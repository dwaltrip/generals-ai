#!/usr/bin/env -S uv run python
"""Run an eval session: N policies play games, with full metrics and replay saves.

Usage (from repo root):

    # Model vs EvalBot, 10 games
    ./scripts/eval_run.py \\
        --policy checkpoint:path/to/epoch_005.pt:force_move=true \\
        --policy evalbot \\
        --games 10

    # Two checkpoints head-to-head
    ./scripts/eval_run.py \\
        --policy checkpoint:path/to/epoch_003.pt:force_move=true \\
        --policy checkpoint:path/to/epoch_005.pt:force_move=true \\
        --games 20

    # 4-player FFA with mixed policies
    ./scripts/eval_run.py \\
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
import datetime as dt
import json
from pathlib import Path
import random
import sys
import time

import torch

from game_runner.policy import GameResult
from game_runner.runner import run_game
from game_runner.save import write_eval_game
from game_runner.seed_map import list_two_player_replay_ids, load_static_from_db
from self_play.agent import default_device


# scripts/eval/ is a sibling directory; add scripts/ to path so we can
# import from the eval package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.metrics_collector import MetricsCollector
from eval.policy_spec import build_policy_names, parse_policy_spec


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
        candidates = list_two_player_replay_ids(limit=200)
        if not candidates:
            raise RuntimeError("no replay maps found in corpus")
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


def run(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    policy_specs: list[str] = args.policy
    num_players = len(policy_specs)
    policy_names = build_policy_names(policy_specs)

    print(f"device: {device}")
    print(f"policies ({num_players} players):")
    for i, name in enumerate(policy_names):
        print(f"  p{i}: {name}")

    # Validate policy specs early (before loading maps or creating output dirs)
    try:
        for i, spec in enumerate(policy_specs):
            parse_policy_spec(spec, slot=i, device=device)
    except (ValueError, FileNotFoundError) as e:
        print(f"\nerror: invalid --policy spec: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate --swap-slots
    swap_slots = args.swap_slots
    if swap_slots and num_players != 2:
        print(
            f"\nerror: --swap-slots only supported for 2-player games, "
            f"got {num_players} policies",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve maps
    replay_ids = _resolve_maps(args.maps, args.map_seed, num_players)
    rounds_per_map = args.games_per_map * (2 if swap_slots else 1)
    total_games = len(replay_ids) * rounds_per_map
    print(f"maps: {len(replay_ids)} unique, {args.games_per_map} games each"
          f"{' (x2 with slot swap)' if swap_slots else ''} = {total_games} total")
    print(f"max_turns: {args.max_turns}")

    # Validate player count against first map
    import sim_core
    static_check = load_static_from_db(replay_ids[0])
    state_check = sim_core.new_state(static_check)
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
    print(f"output: {out_dir}")
    print()

    results_path = out_dir / "results.jsonl"
    win_counts = [0] * num_players
    draw_count = 0
    total_length = 0
    t0 = time.perf_counter()
    game_idx = 0

    swap_rounds = [False, True] if swap_slots else [False]

    for replay_id in replay_ids:
        static = load_static_from_db(replay_id)

        for _rep in range(args.games_per_map):
            for swapped in swap_rounds:
                game_idx += 1
                label = f"game_{game_idx:03d}"
                slot_map = _make_slot_map(num_players, swapped)

                # Build policies in slot order
                policies = [
                    parse_policy_spec(policy_specs[slot_map[s]], slot=s, device=device)
                    for s in range(num_players)
                ]

                # Run with metrics collection
                collector = MetricsCollector(
                    num_players=num_players,
                    sample_interval=args.sample_interval,
                )
                result = run_game(
                    policies, static,
                    max_turns=args.max_turns,
                    on_tick=collector.on_tick,
                )
                metrics = collector.finalize(result, policies)
                total_length += result.game_length

                # Track wins by original policy index
                winner_policy = _winning_policy(result.winner, slot_map)
                if winner_policy is not None:
                    win_counts[winner_policy] += 1
                else:
                    draw_count += 1

                # Save game files
                assert result.state is not None
                write_eval_game(result.state, static, games_dir / f"{label}.npz")
                meta = _build_game_meta(
                    game_idx, replay_id, policy_specs, args.max_turns, slot_map,
                )
                (games_dir / f"{label}.meta.json").write_text(json.dumps(meta, indent=2))
                (games_dir / f"{label}.metrics.json").write_text(json.dumps(metrics, indent=2))

                # Append to results JSONL
                row = _build_results_row(
                    game_idx, replay_id, result, slot_map=slot_map, metrics=metrics,
                )
                with open(results_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

                # Progress line
                slot_names = _names_for_slots(policy_names, slot_map)
                if winner_policy is not None:
                    winner_str = policy_names[winner_policy]
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
