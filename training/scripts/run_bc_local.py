#!/usr/bin/env -S uv run python
"""Local CLI entry point for BC training runs.

Thin wrapper: supplies environment-specific path defaults, then hands off
to `bc.train.bc_run`. The same wiring shape is used by the cloud
entry point (`run_bc_modal.py`).

Run from anywhere in the repo:
    training/scripts/run_bc_local.py --manifest <path> --epochs 1 --max-batches 5
"""

from __future__ import annotations

from pathlib import Path

from bc.resume import bc_resume
from bc.run_dir import initialize_run_dir
from bc.train import bc_run
from bc.train_cli import build_arg_parser, config_from_args, training_overrides


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_INTERMEDIATE = REPO_ROOT / "replay-parser" / "data" / "intermediate"
LOCAL_RUNS_DIR = REPO_ROOT / "training" / "data" / "runs"


def main() -> None:
    parser = build_arg_parser()
    parser.set_defaults(
        intermediate=LOCAL_INTERMEDIATE,
        out_dir=LOCAL_RUNS_DIR,
        device="mps",
    )
    args = parser.parse_args()
    config = config_from_args(args)
    if args.resume:
        bc_resume(config, args.force_config_mismatch, training_overrides(args))
    else:
        initialize_run_dir(config)
        bc_run(config)


if __name__ == "__main__":
    main()
