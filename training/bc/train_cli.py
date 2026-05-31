"""CLI scaffolding for BC training.

The operator supplies a run's reproducible identity — model `arch` + recipe +
data paths — through a single `--config <json>` file; the CLI keeps only
*operational* (invocation-local) knobs: run dir, device, DataLoader knobs,
logging, smoke limits, plus the resume directives. The two domains are disjoint,
so `TrainConfig.from_file` (file) and the operational CLI overrides merge as a
plain union.

Operational knob flags default to `argparse.SUPPRESS`, so a flag that isn't
passed never lands on the namespace and falls through to the `TrainConfig` field
default. `--out-dir` and `--device` are env-wiring the wrappers
(`scripts/run_bc_local.py`, `scripts/run_bc_modal.py`) fill via
`parser.set_defaults(...)` before parsing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bc.train_config import TrainConfig, make_run_id


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=None,
        help=(
            "JSON config file: model `arch` + recipe (lr, batch_size, epochs, "
            "seed, precision, shuffle_buffer_size, gpu) + data paths (manifest, "
            "intermediate). May be partial — unset fields use defaults. On "
            "--resume, this is an overlay applied over the parent's resolved config."
        ),
    )
    # `--out-dir` is a required *value*, but the parser doesn't enforce
    # `required=True` so wrappers can supply an environment default via
    # `set_defaults`. `config_from_args` raises if it's still missing.
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Parent dir for runs. Joined with a fresh run-id to form `TrainConfig.run_dir`.",
    )

    # Operational (invocation-local) knobs. No defaults here — `TrainConfig`'s
    # field defaults are the single source. `default=SUPPRESS` keeps an unpassed flag
    # off the namespace, so `operational_overrides` overlays only what was passed.
    # `--device` is env-wiring the wrappers fill via `set_defaults`.
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--log-every", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "Stop each epoch early after N batches — for smoke testing the "
            "loop end-to-end without committing to a full epoch's runtime."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=argparse.SUPPRESS,
        help="DataLoader worker subprocesses. 0 = main-process iteration.",
    )
    parser.add_argument(
        "--pin-memory",
        choices=("auto", "true", "false"),
        default=argparse.SUPPRESS,
        help="DataLoader pin_memory. 'auto' = True iff device is CUDA.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=argparse.SUPPRESS,
        help="Batches each worker prefetches ahead. Ignored when num-workers=0.",
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip per-epoch val pass. For perf-testing runs where it isn't needed.",
    )

    # Resume directives — not TrainConfig fields. Read by the wrapper to
    # dispatch bc_resume vs. a fresh bc_run.
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume run-id (the timestamp dir name under --out-dir) from its "
            "latest checkpoint. The parent run's config carries over; pass "
            "--config <overlay.json> to change recipe knobs (e.g. raise epochs)."
        ),
    )
    parser.add_argument(
        "--force-config-mismatch",
        action="store_true",
        default=False,
        help=(
            "On --resume, allow trajectory-critical config changes vs. the "
            "parent run (arch/lr/weight_decay still cannot change)."
        ),
    )
    parser.add_argument(
        "--legacy-lr-warmup-batches",
        type=int,
        default=None,
        help=(
            "On --resume from a legacy bare-state_dict checkpoint (no saved "
            "optimizer state), linearly ramp the LR from ~0 to the target over "
            "the first N batches of the resumed segment, while AdamW's variance "
            "estimate re-warms. Required to resume a legacy checkpoint; rejected "
            "on a combined-format one."
        ),
    )
    return parser


_PIN_MEMORY_MAP: dict[str, bool | None] = {"auto": None, "true": True, "false": False}

# argparse dest -> TrainConfig field name (operational knobs). Names match.
_OPERATIONAL_FIELDS: tuple[str, ...] = (
    "device", "log_every", "max_batches",
    "num_workers", "pin_memory", "prefetch_factor", "skip_val",
)


def operational_overrides(args: argparse.Namespace) -> dict[str, object]:
    """The operational (invocation-local) knobs the operator explicitly passed
    (plus `device`, which the wrappers always set). The fresh path overlays these
    onto the config-file values; the resume path overlays them onto the parent
    run's config. Only `pin_memory` needs translating ('auto'/'true'/'false')."""
    vals = vars(args)
    return {
        field: (_PIN_MEMORY_MAP[vals[field]] if field == "pin_memory" else vals[field])
        for field in _OPERATIONAL_FIELDS
        if field in vals
    }


def load_config_overlay(args: argparse.Namespace) -> dict:
    """The `--config` file as a dict (arch + recipe), or `{}` when none is passed.
    Used as the resume overlay over the parent's resolved config."""
    if args.config is None:
        return {}
    return json.loads(Path(args.config).read_text())


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Build a fresh-run `TrainConfig` from `--config` + operational CLI flags.

    `--config` supplies the arch + recipe (over defaults); operational flags
    overlay the invocation-local knobs. `run_dir` is `out_dir/<fresh run-id>`.
    Missing `--config`/`--out-dir`, or resume-only directives passed without
    `--resume`, raise `SystemExit`.
    """
    if args.config is None:
        raise SystemExit("--config is required (or set a default in the wrapper)")
    if args.out_dir is None:
        raise SystemExit("--out-dir is required (or set a default in the wrapper)")
    if args.force_config_mismatch:
        raise SystemExit("--force-config-mismatch only applies with --resume")
    if args.legacy_lr_warmup_batches is not None:
        raise SystemExit("--legacy-lr-warmup-batches only applies with --resume")

    run_dir = args.out_dir / make_run_id()
    return TrainConfig.from_file(args.config, run_dir=run_dir, **operational_overrides(args))


def resume_run_dir(args: argparse.Namespace) -> Path:
    """The run dir a `--resume` targets: `out_dir/<resume-id>` (an existing dir)."""
    if args.out_dir is None:
        raise SystemExit("--out-dir is required (or set a default in the wrapper)")
    return args.out_dir / args.resume
