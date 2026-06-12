"""Orchestrates an eval run described by an `EvalRunSpec`: resolves maps and
policies, plays the games through the batched runner, writes the run dir
(config.json, results.jsonl, per-game replay/meta/metrics artifacts), and
runs the post-hoc analysis over it.

Entry point is `run_eval`; it raises `EvalConfigError` on an unrunnable spec.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
import datetime as dt
import json
from pathlib import Path
import random
import time
import traceback

import torch

from eval_tools.map_set import load_bucket
from eval_tools.metrics_collector import MetricsCollector
from eval_tools.policy_spec import build_policy_names, parse_policy_spec
from eval_tools.run_analysis.pipeline import analyze_run
from eval_tools.run_spec import EvalConfigError, EvalRunSpec
from game_runner.batched import FinishedGame, PendingGame, run_batched
from game_runner.policy import GameResult
from game_runner.save import write_eval_game
from game_runner.seed_map import list_replay_ids_by_player_count, load_static_from_db
from game_types import StaticMap
from training.bc.inference import default_device
from utils.json_io import write_json
from utils.log import tee_stdio


# Default GPU batch width to aim for when --concurrent-games is auto: the pool
# size is derived so concurrent_games × NN-per-game ≈ this. ~24-32 is the M1
# sweet spot; a cloud GPU's optimum is far larger (set --concurrent-games).
_TARGET_NN_ROWS = 32


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return default_device()
    return torch.device(name)


def resolve_maps(
    maps_arg: str, map_seed: int | None, num_players: int,
    map_sets_root: str | None = None,
) -> list[tuple[str, StaticMap]]:
    """Resolve the maps spec to loaded (replay_id, StaticMap) pairs.

    `random:`/`replay_id:` read the collector DB (local-only); `set:` reads a
    frozen map-set pack (works anywhere, and seeded sampling within it is
    stable across time because the set's order is frozen).
    """
    parts = maps_arg.split(":", maxsplit=1)
    mode = parts[0]

    if mode == "random":
        count = int(parts[1]) if len(parts) > 1 else 10
        # NOTE: pulls every matching map from the corpus (no cap). For 8p
        # FFA that's ~180k ids; the list lives in memory only during this
        # function call, and seeded rng.choice over the full list gives
        # unbiased map sampling (with replacement; the corpus grows, so the
        # same seed is only stable at a fixed corpus snapshot).
        candidates = list_replay_ids_by_player_count(num_players)
        if not candidates:
            raise RuntimeError(
                f"no {num_players}-player replay maps found in corpus"
            )
        rng = random.Random(map_seed)
        ids = [rng.choice(candidates) for _ in range(count)]
        return [(rid, load_static_from_db(rid)) for rid in ids]

    elif mode == "replay_id":
        if len(parts) < 2 or not parts[1]:
            raise ValueError("--maps replay_id:id1,id2,... requires at least one ID")
        return [(rid, load_static_from_db(rid)) for rid in parts[1].split(",")]

    elif mode == "set":
        if len(parts) < 2 or not parts[1]:
            raise ValueError("--maps set:<name>[:sample=K] requires a set name")
        name, _, opt = parts[1].partition(":")
        root_kwargs = {"root": Path(map_sets_root)} if map_sets_root else {}
        entries = load_bucket(name, num_players, **root_kwargs)
        if opt:
            key, _, val = opt.partition("=")
            if key != "sample" or not val.isdigit():
                raise ValueError(f"unknown set option '{opt}', expected 'sample=K'")
            k = int(val)
            if k > len(entries):
                raise ValueError(
                    f"sample={k} exceeds the {len(entries)}-map "
                    f"{num_players}p bucket of set '{name}'"
                )
            # Without replacement: a sample of a frozen set is itself a fixed
            # map list, reproducible from (set, K, map_seed) forever.
            entries = random.Random(map_seed).sample(entries, k)
        return entries

    else:
        raise ValueError(
            f"unknown --maps mode '{mode}', expected 'random', 'replay_id', or 'set'"
        )


def build_run_config(spec: EvalRunSpec) -> dict:
    return asdict(spec) | {"timestamp": dt.datetime.now().isoformat()}


def _build_game_meta(
    game_idx: int,
    replay_id: str,
    policy_specs: list[str],
    max_turns: int,
    slot_map: list[int],
) -> dict:
    return {
        "game_index": game_idx,
        "replay_id": replay_id,
        "policy_specs": policy_specs,
        "max_turns": max_turns,
        "slot_map": slot_map,
    }


def _build_results_row(
    game_idx: int,
    replay_id: str,
    result: GameResult,
    slot_map: list[int],
) -> dict:
    """Run-level index row. Metrics live in games/<id>.metrics.json, not here,
    so a scan over outcomes doesn't pay for every game's diagnostics."""
    return {
        "game_index": game_idx,
        "replay_id": replay_id,
        "slot_map": slot_map,
        "winner": result.winner,
        "game_length": result.game_length,
        "player_stats": [
            {
                "land": ps.land,
                "army": ps.army,
                "n_moved": ps.n_moved,
                "n_passed": ps.n_passed,
            }
            for ps in result.player_stats
        ],
    }


def make_slot_map(num_players: int, offset: int) -> list[int]:
    """slot_map[slot] = original policy index for that slot. `offset` cyclically
    rotates the assignment: slot s runs policy (s + offset) % N (offset 0 =
    identity; offset 1 on 2 players = the swap)."""
    return [(s + offset) % num_players for s in range(num_players)]


def _names_for_slots(policy_names: list[str], slot_map: list[int]) -> list[str]:
    """Policy names in slot order for a given game."""
    return [policy_names[slot_map[s]] for s in range(len(slot_map))]


def _winning_policy(winner_slot: int | None, slot_map: list[int]) -> int | None:
    """Map a slot-index winner back to the original policy index."""
    if winner_slot is None:
        return None
    return slot_map[winner_slot]


def rotation_offsets(
    policy_specs: list[str], num_players: int, k: int, skip_dupes: bool,
) -> tuple[list[int], str | None]:
    """Resolve the slot-rotation offsets for K cyclic rotations.

    Each map is played once per offset; offset `o` puts policy (s+o)%N in slot s.
    Offsets are evenly spaced (step N/K), so K=2 lands on the antithetic
    complement pair and K=N fully balances every slot — hence the K | N rule.

    Two offsets collide when identical checkpoints rotate onto the same slots
    (the interleaved A B A B case: rotating by N/2 is a no-op). The dup check
    keys on the raw spec string — two copies of one checkpoint are equal — and
    by default raises; with skip_dupes it drops the duplicate and notes it.

    Returns (offsets_to_run, note). `note` is a line to log when dupes were
    skipped, else None. Raises ValueError (caught up-front by the caller) on a
    bad K or an un-skipped collision.
    """
    if k < 1 or num_players % k != 0:
        divisors = ", ".join(str(d) for d in range(1, num_players + 1) if num_players % d == 0)
        raise ValueError(
            f"--slot-rotations {k} must be a positive divisor of the player "
            f"count ({num_players}); valid values: {divisors}"
        )
    step = num_players // k
    offsets = [i * step for i in range(k)]

    # Label distinct checkpoints A, B, C, ... by first appearance so a reported
    # lineup reads like the whiteboard notation (A A B B vs A B A B).
    letters: dict[str, str] = {}
    for spec in policy_specs:
        letters.setdefault(spec, chr(ord("A") + len(letters)))

    def lineup(o: int) -> tuple[str, ...]:
        return tuple(policy_specs[(s + o) % num_players] for s in range(num_players))

    def label(o: int) -> str:
        return " ".join(letters[policy_specs[(s + o) % num_players]] for s in range(num_players))

    seen: dict[tuple[str, ...], int] = {}
    kept: list[int] = []
    dropped: list[tuple[int, int]] = []
    for o in offsets:
        key = lineup(o)
        if key in seen:
            if not skip_dupes:
                raise ValueError(
                    f"--slot-rotations {k}: offset {o} produces the same slot "
                    f"lineup as offset {seen[key]}:\n"
                    f"    {label(o)}\n"
                    f"Group identical checkpoints contiguously (A A B B, not "
                    f"A B A B), or pass --skip-dupes to run only the distinct lineups."
                )
            dropped.append((o, seen[key]))
            continue
        seen[key] = o
        kept.append(o)

    note = None
    if dropped:
        drops = ", ".join(f"{o}≡{d}" for o, d in dropped)
        noun = "lineup" if len(kept) == 1 else "lineups"
        note = (f"slot-rotations: {k} requested, {len(dropped)} duplicate "
                f"(offsets {drops}) → running {len(kept)} distinct {noun}")
    return kept, note


@dataclass
class _RunCtx:
    """Per-run configuration that doesn't change between games."""
    device: torch.device
    num_players: int
    policy_specs: list[str]
    policy_names: list[str]
    max_turns: int
    sample_interval: int
    games_dir: Path
    results_path: Path


@dataclass
class _GameOutcome:
    """Stat-bearing result of one game, used by the caller to accumulate."""
    winner_policy: int | None  # original policy index of winner, or None for draw
    game_length: int


@dataclass
class _GameCtx:
    """Per-game payload carried on a PendingGame and handed back on the matching
    FinishedGame, so we can record artifacts once the batched runner finishes it."""
    replay_id: str
    slot_map: list[int]
    offset: int
    collector: MetricsCollector
    map_data: StaticMap


def _resolve_pool_size(concurrent_games: int, policy_specs: list[str]) -> int:
    """Concurrent games in the pool. 0 → auto: aim for ~_TARGET_NN_ROWS NN rows,
    i.e. pool × (NN policies per game) ≈ target."""
    if concurrent_games > 0:
        return concurrent_games
    nn_per_game = sum(1 for s in policy_specs if not s.lower().startswith("evalbot"))
    return max(1, round(_TARGET_NN_ROWS / max(1, nn_per_game)))


def _pending_games(
    ctx: _RunCtx, maps: list[tuple[str, StaticMap]], games_per_map: int,
    offsets: list[int],
) -> Iterator[PendingGame]:
    """Yield one PendingGame per (map, rep, rotation), building fresh policies +
    a MetricsCollector lazily as the runner pulls each into the pool. game_id is
    the enumeration order (stable per spec, independent of completion order)."""
    game_idx = 0
    for replay_id, map_data in maps:
        for _rep in range(games_per_map):
            for offset in offsets:
                game_idx += 1
                slot_map = make_slot_map(ctx.num_players, offset)
                policies = [
                    parse_policy_spec(ctx.policy_specs[slot_map[s]], slot=s, device=ctx.device)
                    for s in range(ctx.num_players)
                ]
                collector = MetricsCollector(
                    num_players=ctx.num_players, sample_interval=ctx.sample_interval,
                )
                yield PendingGame(
                    game_id=game_idx,
                    map_data=map_data,
                    policies=policies,
                    on_tick=collector.on_tick,
                    ctx=_GameCtx(replay_id, slot_map, offset, collector, map_data),
                )


def _record_finished(ctx: _RunCtx, fin: FinishedGame) -> _GameOutcome:
    """Save one finished game's artifacts, print a progress line, return stats."""
    g: _GameCtx = fin.ctx
    label = f"game_{fin.game_id:03d}"
    result = fin.result
    metrics = g.collector.finalize(result, fin.policies)
    winner_policy = _winning_policy(result.winner, g.slot_map)

    assert result.state is not None
    write_eval_game(result.state, g.map_data, ctx.games_dir / f"{label}.npz")
    meta = _build_game_meta(
        fin.game_id, g.replay_id, ctx.policy_specs, ctx.max_turns, g.slot_map,
    )
    (ctx.games_dir / f"{label}.meta.json").write_text(json.dumps(meta, indent=2))
    write_json(ctx.games_dir / f"{label}.metrics.json", metrics)

    row = _build_results_row(
        fin.game_id, g.replay_id, result, slot_map=g.slot_map,
    )
    with open(ctx.results_path, "a") as f:
        f.write(json.dumps(row) + "\n")

    slot_names = _names_for_slots(ctx.policy_names, g.slot_map)
    winner_str = ctx.policy_names[winner_policy] if winner_policy is not None else "draw"
    rot_tag = f" [rot {g.offset}]" if g.offset else ""
    lands = " ".join(
        f"{slot_names[s]}={ps.land:3d}" for s, ps in enumerate(result.player_stats)
    )
    print(f"  {label}: {winner_str:12s}  len={result.game_length:4d}  {lands}{rot_tag}")

    return _GameOutcome(winner_policy=winner_policy, game_length=result.game_length)


def run_games(
    spec: EvalRunSpec,
    *,
    device: torch.device,
    policy_names: list[str],
    offsets: list[int],
    rotation_note: str | None,
    maps: list[tuple[str, StaticMap]],
    out_dir: Path,
    games_dir: Path,
) -> None:
    policy_specs = spec.policy_specs
    num_players = len(policy_specs)
    total_games = len(maps) * spec.games_per_map * len(offsets)
    pool_size = _resolve_pool_size(spec.concurrent_games, policy_specs)
    nn_per_game = sum(1 for s in policy_specs if not s.lower().startswith("evalbot"))

    print(f"device: {device}")
    print(f"policies ({num_players} players):")
    for i, name in enumerate(policy_names):
        print(f"  p{i}: {name}")
    rot_desc = "" if len(offsets) == 1 else f" × {len(offsets)} slot-rotations"
    print(f"maps: {len(maps)} unique, {spec.games_per_map} games each"
          f"{rot_desc} = {total_games} total")
    if rotation_note:
        print(rotation_note)
    print(f"max_turns: {spec.max_turns}")
    print(f"pool: {pool_size} concurrent games (~{pool_size * nn_per_game} NN rows/forward)")
    print(f"output: {out_dir}")
    print()

    results_path = out_dir / "results.jsonl"
    ctx = _RunCtx(
        device=device,
        num_players=num_players,
        policy_specs=policy_specs,
        policy_names=policy_names,
        max_turns=spec.max_turns,
        sample_interval=spec.sample_interval,
        games_dir=games_dir,
        results_path=results_path,
    )
    win_counts = [0] * num_players
    draw_count = 0
    total_length = 0
    n_done = 0
    t0 = time.perf_counter()

    pending = _pending_games(ctx, maps, spec.games_per_map, offsets)
    # Games finish out of order; each results.jsonl row carries game_index.
    for fin in run_batched(pending, pool_size=pool_size, max_turns=spec.max_turns):
        outcome = _record_finished(ctx, fin)
        n_done += 1
        total_length += outcome.game_length
        if outcome.winner_policy is not None:
            win_counts[outcome.winner_policy] += 1
        else:
            draw_count += 1

    elapsed = time.perf_counter() - t0

    # Summary
    print()
    print(f"completed {n_done} games in {elapsed:.1f}s")
    for p in range(num_players):
        print(f"  {policy_names[p]:20s}: "
              f"{win_counts[p]:3d} wins ({win_counts[p]/n_done:.0%})")
    if draw_count:
        print(f"  {'draws':20s}: {draw_count:3d}     ({draw_count/n_done:.0%})")

    print(f"  avg game length: {total_length / n_done:.0f} ticks")
    print(f"\nresults: {results_path}")
    print(f"replays: {games_dir}")


def run_eval(spec: EvalRunSpec) -> Path:
    """Run the eval described by `spec`; returns the run dir it created."""
    device = resolve_device(spec.device)
    policy_specs = spec.policy_specs
    num_players = len(policy_specs)
    policy_names = build_policy_names(policy_specs)

    # Validate everything up front (no stdout prints): a failure here raises
    # before we create the output dir, so bad invocations don't leave behind
    # empty run dirs.
    try:
        for i, s in enumerate(policy_specs):
            parse_policy_spec(s, slot=i, device=device)
    except (ValueError, FileNotFoundError) as e:
        raise EvalConfigError(f"invalid policy spec: {e}") from e

    try:
        offsets, rotation_note = rotation_offsets(
            policy_specs, num_players, spec.slot_rotations, spec.skip_dupes,
        )
    except ValueError as e:
        raise EvalConfigError(str(e)) from e

    try:
        maps = resolve_maps(spec.maps, spec.map_seed, num_players, spec.map_sets_root)
    except (ValueError, RuntimeError, FileNotFoundError, LookupError) as e:
        raise EvalConfigError(str(e)) from e

    import sim_core
    state_check = sim_core.new_state(maps[0][1])
    if state_check.num_players != num_players:
        raise EvalConfigError(
            f"map has {state_check.num_players} player slots but "
            f"{num_players} policies were specified"
        )

    # Set up output directory. A caller-supplied run_name lets fire-and-forget
    # entry points (cloud spawn) know the run dir without waiting on the result.
    name = spec.run_name or dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = Path(spec.out_root) / name
    games_dir = out_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = build_run_config(spec)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    with tee_stdio(out_dir / "run.log"):
        run_games(
            spec,
            device=device,
            policy_names=policy_names,
            offsets=offsets,
            rotation_note=rotation_note,
            maps=maps,
            out_dir=out_dir,
            games_dir=games_dir,
        )

        # The games are on disk at this point, so an analysis crash shouldn't
        # read as a failed run — report it and leave the post-hoc CLI as the
        # fallback.
        print()
        try:
            analyze_run(out_dir)
        except Exception:
            traceback.print_exc()
            print(f"analysis failed — run "
                  f"./packages/eval-tools/scripts/analyze_eval.py {out_dir} manually")

    return out_dir
