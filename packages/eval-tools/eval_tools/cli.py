"""Shared CLI surface for eval entry points.

`build_arg_parser` defines the adhoc-eval arguments and `spec_from_args`
resolves them into an `EvalRunSpec`. Local (`scripts/run_adhoc.py`) and cloud
(`scripts/run_eval_modal.py`) entry points share both — mirroring how
training's local/modal scripts share `train_cli.build_arg_parser` — so the two
invocations stay argument-compatible.
"""

from __future__ import annotations

import argparse

from eval_tools.run_spec import EvalConfigError, EvalRunSpec


def build_arg_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--policy", action="append", required=True,
        help="Policy spec (one per player slot). "
             "Format: 'checkpoint:path[:key=val,...]' or 'evalbot[:config=path]'",
    )
    parser.add_argument("--maps", type=str, default="random:10",
                        help="Map selection: 'random:N' (corpus-wide, local DB), "
                             "'replay_id:id1,id2,...', or 'set:<name>[:sample=K]' "
                             "(frozen eval map set, e.g. set:eval-map-set-v1:sample=20)")
    parser.add_argument("--map-seed", type=int, default=42,
                        help="RNG seed for random/set map sampling")
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
    return parser


def spec_from_args(args: argparse.Namespace, **overrides) -> EvalRunSpec:
    """Resolve parsed args into an `EvalRunSpec`. `overrides` replace any spec
    field — the cloud entry point uses this for container paths (out_root,
    map_sets_root) and translated policy specs. Raises `EvalConfigError` on
    inconsistent flag combinations."""
    if args.swap_slots and args.slot_rotations not in (1, 2):
        raise EvalConfigError(
            "--swap-slots is an alias for --slot-rotations 2; don't combine "
            f"it with --slot-rotations {args.slot_rotations}"
        )
    fields = {
        "policy_specs": args.policy,
        "maps": args.maps,
        "map_seed": args.map_seed,
        "games_per_map": args.games_per_map,
        "slot_rotations": 2 if args.swap_slots else args.slot_rotations,
        "skip_dupes": args.skip_dupes,
        "max_turns": args.max_turns,
        "device": args.device,
        "sample_interval": args.sample_interval,
        "concurrent_games": args.concurrent_games,
    }
    return EvalRunSpec(**(fields | overrides))
