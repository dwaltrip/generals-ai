#!/usr/bin/env -S uv run python
"""Local CLI entry point for BC training runs.

Thin wrapper: supplies environment-specific defaults (`--out-dir`, `--device`),
then hands off to `bc.train.bc_run` (fresh) or `bc.resume.bc_resume` (resume).
The same wiring shape is used by the cloud entry point (`run_bc_modal.py`).
The run's arch + recipe + data paths come from the `--config` file.

Run from anywhere in the repo:
    training/scripts/run_bc_local.py --config <path> --max-batches 5
"""

from __future__ import annotations

from pathlib import Path

from bc.resume import bc_resume
from bc.run_dir import initialize_run_dir, prepare_resume
from bc.run_instrumentation import instrumented_run
from bc.train import bc_run
from bc.train_cli import (
    build_arg_parser,
    config_from_args,
    load_config_overlay,
    operational_overrides,
    resume_run_dir,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_RUNS_DIR = REPO_ROOT / "training" / "data" / "runs"


def main() -> None:
    parser = build_arg_parser()
    parser.set_defaults(out_dir=LOCAL_RUNS_DIR, device="mps")
    args = parser.parse_args()
    if args.resume:
        run_dir = resume_run_dir(args)
        info = prepare_resume(run_dir)
        with instrumented_run(run_dir, args.profile):
            bc_resume(run_dir, info, load_config_overlay(args), operational_overrides(args),
                      args.force_config_mismatch, args.legacy_lr_warmup_batches)
    else:
        config = config_from_args(args)
        initialize_run_dir(config, Path(args.config).read_text())
        with instrumented_run(config.run_dir, args.profile):
            bc_run(config)


if __name__ == "__main__":
    main()
