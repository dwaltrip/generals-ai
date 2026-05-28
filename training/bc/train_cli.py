"""CLI scaffolding for BC training.

Maps an `argparse.Namespace` into a validated `TrainConfig`. The parser
itself carries no environment-specific path defaults — wrappers
(`scripts/run_bc_local.py`, `scripts/run_bc_modal.py`) supply those via
`parser.set_defaults(...)` before parsing.
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=2048,
        help=(
            "Reservoir-shuffle buffer over the IterableDataset's yielded "
            "samples. Decorrelates batches from the per-perspective causal "
            "walk. Pass 0 to disable (raw walk order)."
        ),
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help=(
            "Stop each epoch early after N batches — for smoke testing the "
            "loop end-to-end without committing to a full epoch's runtime."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker subprocesses. 0 = main-process iteration.",
    )
    parser.add_argument(
        "--pin-memory",
        choices=("auto", "true", "false"),
        default="auto",
        help="DataLoader pin_memory. 'auto' = True iff device is CUDA.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches each worker prefetches ahead. Ignored when num-workers=0.",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16"),
        default="auto",
        help="Mixed-precision mode. 'auto' = fp16 on CUDA, fp32 elsewhere.",
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="Skip the per-epoch val pass. For perf-testing runs where val cost dominates wall time.",
    )
    parser.add_argument(
        "--value-head",
        choices=("direct", "pyramid"),
        default="direct",
        help=(
            "Value-head architecture variant. 'direct' = thin Conv+Linear "
            "(~9k params); 'pyramid' = adds a PyramidModule before projection "
            "for multi-scale receptive field (~+0.82M params)."
        ),
    )
    return parser


_PIN_MEMORY_MAP: dict[str, bool | None] = {"auto": None, "true": True, "false": False}


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Build a `TrainConfig` from parsed CLI args. Raises `SystemExit` if a
    required path is missing (wrapper forgot to set a default, and the user
    didn't pass it)."""
    if args.manifest is None:
        raise SystemExit("--manifest is required (or set a default in the wrapper)")
    if args.intermediate is None:
        raise SystemExit("--intermediate is required (or set a default in the wrapper)")
    if args.out_dir is None:
        raise SystemExit("--out-dir is required (or set a default in the wrapper)")
    return TrainConfig(
        manifest=args.manifest,
        intermediate=args.intermediate,
        run_dir=args.out_dir / make_run_id(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        log_every=args.log_every,
        max_batches=args.max_batches,
        num_workers=args.num_workers,
        pin_memory=_PIN_MEMORY_MAP[args.pin_memory],
        prefetch_factor=args.prefetch_factor,
        precision=args.precision,
        skip_val=args.skip_val,
        value_head_variant=args.value_head,
    )
