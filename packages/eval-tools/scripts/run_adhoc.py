#!/usr/bin/env -S uv run python
"""Run an eval session: N policies play games, with full metrics and replay saves.

Usage (from repo root):

    # Model vs EvalBot, 10 games
    ./packages/eval-tools/scripts/run_adhoc.py \\
        --policy checkpoint:path/to/epoch_005.pt:force_move=true \\
        --policy evalbot \\
        --maps random:10

    # Two checkpoints head-to-head, 20 games
    ./packages/eval-tools/scripts/run_adhoc.py \\
        --policy checkpoint:path/to/epoch_003.pt:force_move=true \\
        --policy checkpoint:path/to/epoch_005.pt:force_move=true \\
        --maps random:10 --games-per-map 2

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
        results.jsonl            # one row per game (result + per-seat stats)
        games/
            game_001.npz         # compressed replay data
            game_001.meta.json   # policy configs, map ID, settings
            game_001.metrics.json  # computed metrics + diagnostics
"""

import argparse
import sys

from eval_tools.run_spec import EvalConfigError, EvalRunSpec
from eval_tools.runner import run_eval


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
    parser.add_argument("--slot-rotations", type=int, default=1, metavar="K",
                        help="Play each map K times, cyclically rotating which policy "
                             "occupies which slot (K must divide the player count). K=1 "
                             "(default) plays once; K=2 is the antithetic complement pair; "
                             "K=<players> fully balances every slot. Identical-checkpoint "
                             "lineups that collide under rotation error out unless "
                             "--skip-dupes is set.")
    parser.add_argument("--swap-slots", action="store_true",
                        help="Alias for --slot-rotations 2 (the antithetic complement "
                             "pair). Requires an even player count.")
    parser.add_argument("--skip-dupes", action="store_true",
                        help="When rotations produce identical lineups (e.g. interleaved "
                             "A B A B), run only the distinct lineups instead of erroring.")
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--sample-interval", type=int, default=25,
                        help="Sample land/army curves every N ticks (0 to disable)")
    parser.add_argument("--concurrent-games", type=int, default=0,
                        help="Games run concurrently in the batched pool. "
                             "0 = auto (~32 NN rows/forward). GPU batch ≈ this × NN-per-game.")
    args = parser.parse_args()

    if args.swap_slots and args.slot_rotations not in (1, 2):
        parser.error(
            "--swap-slots is an alias for --slot-rotations 2; don't combine "
            f"it with --slot-rotations {args.slot_rotations}"
        )

    spec = EvalRunSpec(
        policy_specs=args.policy,
        maps=args.maps,
        map_seed=args.map_seed,
        games_per_map=args.games_per_map,
        slot_rotations=2 if args.swap_slots else args.slot_rotations,
        skip_dupes=args.skip_dupes,
        max_turns=args.max_turns,
        device=args.device,
        sample_interval=args.sample_interval,
        concurrent_games=args.concurrent_games,
    )
    try:
        run_eval(spec)
    except EvalConfigError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
