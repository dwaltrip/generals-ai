"""CLI scaffolding for BC training.

Maps an `argparse.Namespace` into a validated `TrainConfig`. The parser
owns no defaults: hyperparameter defaults live on `TrainConfig` (the single
source), and environment-specific paths/device come from the wrappers
(`scripts/run_bc_local.py`, `scripts/run_bc_modal.py`) via
`parser.set_defaults(...)` before parsing.

Training-knob flags default to `argparse.SUPPRESS`, so a flag that isn't
passed never lands on the namespace. `config_from_args` overlays only the
explicitly-passed flags onto `TrainConfig`'s field defaults. A side benefit:
`vars(args)` is then exactly "what the operator passed", which the resume
path uses to overlay a parent run's config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bc.train_config import TrainConfig, make_run_id


def build_arg_parser() -> argparse.ArgumentParser:
    # TODO: support a `--config <file>` JSON input (deferred). With ~18 knobs
    # and cross-run sweeps, a file beats long CLI lines. Read the same schema
    # we already emit as `args.json`, so a run's args.json round-trips as input
    # (reproduce-a-run + resume fall out for free). Precedence:
    #   TrainConfig defaults < --config file < explicit CLI flags < env paths.
    # Hand-rolled merge, not Hydra/OmegaConf (overkill at this scale).
    parser = argparse.ArgumentParser()
    # `--manifest`, `--intermediate`, `--out-dir` are required *values*, but
    # the parser doesn't enforce `required=True` so wrappers can supply
    # environment defaults via `set_defaults` (e.g. the cloud wrapper points
    # `--manifest` at the Volume-mounted probe manifest). `config_from_args`
    # raises if any is still missing after parse.
    parser.add_argument("--manifest", type=Path, default=None)
    # TODO: possibly rename `--intermediate` arg? it's a bit of a vague name.
    parser.add_argument("--intermediate", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Parent dir for runs. Joined with a fresh run-id to form `TrainConfig.run_dir`.")

    # Training knobs carry no defaults here — `TrainConfig`'s field defaults
    # are the single source. `default=SUPPRESS` keeps an unpassed flag off the
    # namespace entirely, so `config_from_args` overlays only what was passed
    # and the rest fall through to the dataclass. `--device` is env-wiring the
    # wrappers fill via `set_defaults` (which overrides SUPPRESS).
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "Reservoir-shuffle buffer over the IterableDataset's yielded "
            "samples. Decorrelates batches from the per-perspective causal "
            "walk. Pass 0 to disable (raw walk order)."
        ),
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
        "--precision",
        choices=("auto", "fp32", "fp16"),
        default=argparse.SUPPRESS,
        help="Mixed-precision mode. 'auto' = fp16 on CUDA, fp32 elsewhere.",
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip the per-epoch val pass. For perf-testing runs where val cost dominates wall time.",
    )
    parser.add_argument(
        "--value-head",
        choices=("direct", "pyramid"),
        default=argparse.SUPPRESS,
        help=(
            "Value-head architecture variant. 'direct' = thin Conv+Linear "
            "(~9k params); 'pyramid' = adds a PyramidModule before projection "
            "for multi-scale receptive field (~+0.82M params)."
        ),
    )

    # Resume directives — not TrainConfig fields. Read by the wrapper to
    # dispatch bc_resume vs. a fresh bc_run, and by config_from_args to point
    # run_dir at the resume target.
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume run-id (the timestamp dir name under --out-dir) from its "
            "latest checkpoint. Unspecified knobs carry over from the parent run."
        ),
    )
    parser.add_argument(
        "--force-config-mismatch",
        action="store_true",
        default=False,
        help=(
            "On --resume, allow trajectory-critical config changes vs. the "
            "parent run (lr/weight_decay still cannot change)."
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

# argparse dest -> TrainConfig field name. Names match the field except for
# `value_head` (the flag) -> `value_head_variant` (the field).
_TRAINING_FIELDS: dict[str, str] = {
    "epochs": "epochs",
    "batch_size": "batch_size",
    "lr": "lr",
    "weight_decay": "weight_decay",
    "device": "device",
    "seed": "seed",
    "shuffle_buffer_size": "shuffle_buffer_size",
    "log_every": "log_every",
    "max_batches": "max_batches",
    "num_workers": "num_workers",
    "pin_memory": "pin_memory",
    "prefetch_factor": "prefetch_factor",
    "precision": "precision",
    "skip_val": "skip_val",
    "value_head": "value_head_variant",
}


def _adapt(dest: str, value: object) -> object:
    """Translate a raw CLI value to its `TrainConfig` form. Only `pin_memory`
    needs it ('auto'/'true'/'false' -> None/True/False); everything else
    passes through unchanged."""
    if dest == "pin_memory":
        return _PIN_MEMORY_MAP[value]
    return value


def training_overrides(args: argparse.Namespace) -> dict[str, object]:
    """The training-knob fields the operator explicitly passed — the resume
    overlay set. Same projection `config_from_args` applies before filling the
    rest from `TrainConfig` defaults; the resume path overlays it onto the
    parent run's config instead."""
    return {
        _TRAINING_FIELDS[dest]: _adapt(dest, val)
        for dest, val in vars(args).items()
        if dest in _TRAINING_FIELDS
    }


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Build a `TrainConfig` from parsed CLI args.

    Required paths (`--manifest`/`--intermediate`/`--out-dir`) are read
    directly; a missing one raises `SystemExit` (wrapper forgot a default and
    the user didn't pass it). Training knobs are overlaid onto `TrainConfig`'s
    field defaults — only flags the operator actually passed appear in
    `vars(args)` (the rest defaulted to `SUPPRESS`), so unspecified knobs fall
    through to the dataclass default.

    `run_dir` is `out_dir/<run-id>`; with `--resume` the run-id is the resume
    target (an existing dir) rather than a fresh timestamp."""
    if args.manifest is None:
        raise SystemExit("--manifest is required (or set a default in the wrapper)")
    if args.intermediate is None:
        raise SystemExit("--intermediate is required (or set a default in the wrapper)")
    if args.out_dir is None:
        raise SystemExit("--out-dir is required (or set a default in the wrapper)")
    if args.force_config_mismatch and not args.resume:
        raise SystemExit("--force-config-mismatch only applies with --resume")
    if args.legacy_lr_warmup_batches is not None:
        if not args.resume:
            raise SystemExit("--legacy-lr-warmup-batches only applies with --resume")
        if args.legacy_lr_warmup_batches < 1:
            raise SystemExit("--legacy-lr-warmup-batches must be >= 1")

    return TrainConfig(
        manifest=args.manifest,
        intermediate=args.intermediate,
        run_dir=args.out_dir / (args.resume or make_run_id()),
        **training_overrides(args),
    )
