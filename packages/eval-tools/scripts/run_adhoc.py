#!/usr/bin/env -S uv run python
"""Run an eval session: N policies play games, with full metrics and replay saves.

Usage (from repo root):

    # Model vs EvalBot, 10 games
    ./packages/eval-tools/scripts/run_adhoc.py \\
        --policy checkpoint:path/to/epoch_005.pt:force_move=true \\
        --policy evalbot \\
        --games 10

    # Two checkpoints head-to-head
    ./packages/eval-tools/scripts/run_adhoc.py \\
        --policy checkpoint:path/to/epoch_003.pt:force_move=true \\
        --policy checkpoint:path/to/epoch_005.pt:force_move=true \\
        --games 20

    # 4-player FFA with mixed policies
    ./packages/eval-tools/scripts/run_adhoc.py \\
        --policy checkpoint:path/to/model.pt \\
        --policy evalbot \\
        --policy evalbot \\
        --policy evalbot \\
        --maps random:5 --games-per-map 3

Output structure:
    data/eval-runs/<timestamp>/
        config.json              # full run config (reproducible)
        results.jsonl            # one row per game (result + metrics)
        games/
            game_001.npz         # compressed replay data
            game_001.meta.json   # policy configs, map ID, settings
            game_001.metrics.json  # computed metrics + diagnostics
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
import random
import sys
import time

from bc.inference import default_device
from game_types import StaticMap
import torch

from eval_tools.metrics_collector import MetricsCollector
from eval_tools.policy_spec import build_policy_names, parse_policy_spec
from game_runner.policy import GameResult
from game_runner.runner import run_game
from game_runner.save import write_eval_game
from game_runner.seed_map import list_replay_ids_by_player_count, load_static_from_db
from utils.log import tee_stdio


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return default_device()
    return torch.device(name)


def _resolve_maps(
    maps_arg: str, map_seed: int | None, num_players: int,
) -> list[str]:
    parts = maps_arg.split(":", maxsplit=1)
    mode = parts[0]

    if mode == "random":
        count = int(parts[1]) if len(parts) > 1 else 10
        # NOTE: pulls every matching map from the corpus (no cap). For 8p
        # FFA that's ~180k ids; the list lives in memory only during this
        # function call, and seeded rng.choice over the full list gives
        # unbiased map sampling.
        candidates = list_replay_ids_by_player_count(num_players)
        if not candidates:
            raise RuntimeError(
                f"no {num_players}-player replay maps found in corpus"
            )
        rng = random.Random(map_seed)
        return [rng.choice(candidates) for _ in range(count)]

    elif mode == "replay_id":
        if len(parts) < 2 or not parts[1]:
            raise ValueError("--maps replay_id:id1,id2,... requires at least one ID")
        return parts[1].split(",")

    else:
        raise ValueError(f"unknown --maps mode '{mode}', expected 'random' or 'replay_id'")


def _build_run_config(args: argparse.Namespace) -> dict:
    return {
        "policy_specs": args.policy,
        "maps_arg": args.maps,
        "map_seed": args.map_seed,
        "games_per_map": args.games_per_map,
        "swap_slots": args.swap_slots,
        "max_turns": args.max_turns,
        "device": args.device,
        "sample_interval": args.sample_interval,
        "timestamp": dt.datetime.now().isoformat(),
    }


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
    metrics: dict,
    slot_map: list[int],
) -> dict:
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
        "metrics": metrics,
    }


def _make_slot_map(num_players: int, swapped: bool) -> list[int]:
    """slot_map[slot] = original policy index for that slot."""
    identity = list(range(num_players))
    return list(reversed(identity)) if swapped else identity


def _names_for_slots(policy_names: list[str], slot_map: list[int]) -> list[str]:
    """Policy names in slot order for a given game."""
    return [policy_names[slot_map[s]] for s in range(len(slot_map))]


def _winning_policy(winner_slot: int | None, slot_map: list[int]) -> int | None:
    """Map a slot-index winner back to the original policy index."""
    if winner_slot is None:
        return None
    return slot_map[winner_slot]


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


def _play_and_record_game(
    ctx: _RunCtx,
    game_idx: int,
    replay_id: str,
    map_data: StaticMap,
    slot_map: list[int],
    swapped: bool,
) -> _GameOutcome:
    """Play one configured game, save its artifacts, print a progress line."""
    label = f"game_{game_idx:03d}"

    policies = [
        parse_policy_spec(ctx.policy_specs[slot_map[s]], slot=s, device=ctx.device)
        for s in range(ctx.num_players)
    ]

    collector = MetricsCollector(
        num_players=ctx.num_players,
        sample_interval=ctx.sample_interval,
    )
    result = run_game(
        policies, map_data,
        max_turns=ctx.max_turns,
        on_tick=collector.on_tick,
    )
    metrics = collector.finalize(result, policies)

    winner_policy = _winning_policy(result.winner, slot_map)

    assert result.state is not None
    write_eval_game(result.state, map_data, ctx.games_dir / f"{label}.npz")
    meta = _build_game_meta(
        game_idx, replay_id, ctx.policy_specs, ctx.max_turns, slot_map,
    )
    (ctx.games_dir / f"{label}.meta.json").write_text(json.dumps(meta, indent=2))
    (ctx.games_dir / f"{label}.metrics.json").write_text(json.dumps(metrics, indent=2))

    row = _build_results_row(
        game_idx, replay_id, result, slot_map=slot_map, metrics=metrics,
    )
    with open(ctx.results_path, "a") as f:
        f.write(json.dumps(row) + "\n")

    slot_names = _names_for_slots(ctx.policy_names, slot_map)
    if winner_policy is not None:
        winner_str = ctx.policy_names[winner_policy]
    else:
        winner_str = "draw"
    swap_tag = " [swapped]" if swapped else ""
    lands = " ".join(
        f"{slot_names[s]}={ps.land:3d}"
        for s, ps in enumerate(result.player_stats)
    )
    print(
        f"  {label}: {winner_str:12s}  len={result.game_length:4d}"
        f"  {lands}{swap_tag}"
    )

    return _GameOutcome(
        winner_policy=winner_policy,
        game_length=result.game_length,
    )


def run(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    policy_specs: list[str] = args.policy
    num_players = len(policy_specs)
    policy_names = build_policy_names(policy_specs)

    # Validate everything up front (no stdout prints): a failure here exits
    # before we create the output dir, so we don't leave behind empty run
    # dirs on bad invocations. tee_stdio also catches SystemExit and would
    # dump a misleading traceback into the log if we exited inside it.
    try:
        for i, spec in enumerate(policy_specs):
            parse_policy_spec(spec, slot=i, device=device)
    except (ValueError, FileNotFoundError) as e:
        print(f"\nerror: invalid --policy spec: {e}", file=sys.stderr)
        sys.exit(1)

    swap_slots = args.swap_slots
    if swap_slots and num_players != 2:
        print(
            f"\nerror: --swap-slots only supported for 2-player games, "
            f"got {num_players} policies",
            file=sys.stderr,
        )
        sys.exit(1)

    replay_ids = _resolve_maps(args.maps, args.map_seed, num_players)
    rounds_per_map = args.games_per_map * (2 if swap_slots else 1)
    total_games = len(replay_ids) * rounds_per_map

    import sim_core
    map_check = load_static_from_db(replay_ids[0])
    state_check = sim_core.new_state(map_check)
    if state_check.num_players != num_players:
        print(
            f"\nerror: map has {state_check.num_players} player slots but "
            f"{num_players} policies were specified",
            file=sys.stderr,
        )
        sys.exit(1)

    # Set up output directory
    ts = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = Path("data/eval-runs") / ts
    games_dir = out_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = _build_run_config(args)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    with tee_stdio(out_dir / "run.log"):
        _run_games(
            args=args,
            device=device,
            policy_specs=policy_specs,
            policy_names=policy_names,
            num_players=num_players,
            swap_slots=swap_slots,
            replay_ids=replay_ids,
            total_games=total_games,
            out_dir=out_dir,
            games_dir=games_dir,
        )


def _run_games(
    *,
    args: argparse.Namespace,
    device: torch.device,
    policy_specs: list[str],
    policy_names: list[str],
    num_players: int,
    swap_slots: bool,
    replay_ids: list[str],
    total_games: int,
    out_dir: Path,
    games_dir: Path,
) -> None:
    print(f"device: {device}")
    print(f"policies ({num_players} players):")
    for i, name in enumerate(policy_names):
        print(f"  p{i}: {name}")
    print(f"maps: {len(replay_ids)} unique, {args.games_per_map} games each"
          f"{' (x2 with slot swap)' if swap_slots else ''} = {total_games} total")
    print(f"max_turns: {args.max_turns}")
    print(f"output: {out_dir}")
    print()

    results_path = out_dir / "results.jsonl"
    ctx = _RunCtx(
        device=device,
        num_players=num_players,
        policy_specs=policy_specs,
        policy_names=policy_names,
        max_turns=args.max_turns,
        sample_interval=args.sample_interval,
        games_dir=games_dir,
        results_path=results_path,
    )
    win_counts = [0] * num_players
    draw_count = 0
    total_length = 0
    t0 = time.perf_counter()
    game_idx = 0

    swap_rounds = [False, True] if swap_slots else [False]

    for replay_id in replay_ids:
        map_data = load_static_from_db(replay_id)
        for _rep in range(args.games_per_map):
            for swapped in swap_rounds:
                game_idx += 1
                slot_map = _make_slot_map(num_players, swapped)
                outcome = _play_and_record_game(
                    ctx, game_idx, replay_id, map_data, slot_map, swapped,
                )
                total_length += outcome.game_length
                if outcome.winner_policy is not None:
                    win_counts[outcome.winner_policy] += 1
                else:
                    draw_count += 1

    elapsed = time.perf_counter() - t0

    # Summary
    print()
    print(f"completed {game_idx} games in {elapsed:.1f}s")
    for p in range(num_players):
        print(f"  {policy_names[p]:20s}: "
              f"{win_counts[p]:3d} wins ({win_counts[p]/game_idx:.0%})")
    if draw_count:
        print(f"  {'draws':20s}: {draw_count:3d}     ({draw_count/game_idx:.0%})")

    print(f"  avg game length: {total_length / game_idx:.0f} ticks")
    print(f"\nresults: {results_path}")
    print(f"replays: {games_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--policy", action="append", required=True,
        help="Policy spec (one per player slot). "
             "Format: 'checkpoint:path[:key=val,...]' or 'evalbot[:config=path]'",
    )
    parser.add_argument("--maps", type=str, default="random:10",
                        help="Map selection: 'random:N' or 'replay_id:id1,id2,...'")
    parser.add_argument("--map-seed", type=int, default=42,
                        help="RNG seed for random map selection")
    parser.add_argument("--games-per-map", type=int, default=1,
                        help="Number of games to play per map")
    parser.add_argument("--swap-slots", action="store_true",
                        help="Play each game twice with slots reversed (2-player only). "
                             "Doubles the total game count.")
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--sample-interval", type=int, default=25,
                        help="Sample land/army curves every N ticks (0 to disable)")
    args = parser.parse_args()

    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
