#!/usr/bin/env -S uv run python
"""Run an eval session: N policies play games, with full metrics and replay saves.

Usage (from repo root):

    # Model vs EvalBot, 10 games
    ./packages/eval-tools/scripts/run_adhoc.py \\
        --policy checkpoint:path/to/epoch_005.pt:force_move=true \\
        --policy evalbot \\
        --maps random:10

    # Two checkpoints head-to-head on the frozen eval map set
    ./packages/eval-tools/scripts/run_adhoc.py \\
        --policy checkpoint:path/to/epoch_003.pt:force_move=true \\
        --policy checkpoint:path/to/epoch_005.pt:force_move=true \\
        --maps set:eval-map-set-v1:sample=20 --games-per-map 2

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

import sys

from eval_tools.cli import build_arg_parser, spec_from_args
from eval_tools.run_spec import EvalConfigError
from eval_tools.runner import run_eval


def main() -> int:
    parser = build_arg_parser(description=__doc__)
    args = parser.parse_args()
    try:
        spec = spec_from_args(args)
        run_eval(spec)
    except EvalConfigError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
