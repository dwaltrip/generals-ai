#!/usr/bin/env -S uv run python
"""Frozen-representation probe CLI.

Trains small swap-in heads on a frozen representation (a trained trunk, or the
raw obs as a control) to read whether a target signal is decodable, and at what
capacity. Built for the who-dies-next post-mortem (docs 6.13-15 / 6.14): does a
frozen trunk still let a linear readout recover the 47% lowest-army rule, or did
training for policy+value(+elim) discard a signal the obs hands over directly?

The matrix that gives the firm read is run as separate invocations sharing a
task — one per representation:

    # this trunk (elim-ON), all head capacities:
    trunk_probe.py --task lowest_army --source trunk \\
        --checkpoint <sweep>/runs/<id>/checkpoints/epoch_002.pt --manifest <11k split>

    # the sanity anchor — same target read straight off the obs:
    trunk_probe.py --task lowest_army --source raw_obs \\
        --checkpoint <same ckpt, for obs config> --manifest <11k split>

    # insight-4 redundancy: a policy+value-only trunk:
    trunk_probe.py --task lowest_army --source trunk \\
        --checkpoint <elim-OFF base ckpt> --manifest <11k split>

Read linear_pool vs. fat across the rows: raw_obs near the rule everywhere is
the sanity pass; trunk linear ≈ rule → preserved (the trained head's miss is
training, not representation); only fat → entangled; neither → mangled.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import torch

from settings import INTERMEDIATE_DIR
from training.analysis.probes.core import (
    RawObsSource,
    TrunkSource,
    load_frozen_model,
    run_probe,
)
from training.analysis.probes.tasks import TASKS
from training.bc.splits import load_manifest, samples_for_split
from training.settings import PROBES_DIR
from training.shared.device import disable_mps_fallback, pick_device
from utils.docstring import doc_summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=doc_summary(__doc__))
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--source", choices=("trunk", "raw_obs"), default="trunk")
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Trunk checkpoint. For --source raw_obs, used only to read the "
        "obs config so the control matches the trunk run's inputs.",
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--intermediate", type=Path, default=INTERMEDIATE_DIR)
    p.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--cache-chunk", type=int, default=64)
    p.add_argument(
        "--max-train-frames", type=int, default=None,
        help="Cap train_probe frames (smoke runs). Default: all.",
    )
    p.add_argument("--max-val-frames", type=int, default=None)
    p.add_argument("--out-root", type=Path, default=PROBES_DIR)
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    disable_mps_fallback()

    for path in (args.checkpoint, args.manifest, args.intermediate):
        if not path.exists():
            raise SystemExit(f"not found: {path}")

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device}")

    # The trunk checkpoint is authoritative for the obs config: the control
    # must read the same obs tensor the trunk was trained on.
    model = load_frozen_model(args.checkpoint, device)
    obs_cfg = model.cfg.obs
    print(f"obs config: dense_history_n={obs_cfg.dense_history_n} "
          f"({obs_cfg.obs_channels} channels)")

    if args.source == "trunk":
        source = TrunkSource(model.trunk, out_channels=model.cfg.outer_width)
    else:
        source = RawObsSource(out_channels=obs_cfg.obs_channels)

    task = TASKS[args.task](dense_history_n=obs_cfg.dense_history_n)

    manifest = load_manifest(args.manifest)
    val_samples = samples_for_split(manifest, "val", args.intermediate)
    print(f"manifest: {args.manifest.name}  ({len(val_samples)} val perspectives)")

    out_dir = args.out_root / f"{datetime.now():%Y%m%d-%H%M%S}-{args.task}-{args.source}"
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"out dir: {out_dir}\n")

    results = run_probe(
        task, source, val_samples, obs_cfg,
        device=device, seed=args.seed, val_frac=args.val_frac,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        cache_chunk=args.cache_chunk,
        max_train_frames=args.max_train_frames, max_val_frames=args.max_val_frames,
    )

    summary = {
        "task": args.task,
        "source": args.source,
        "checkpoint": str(args.checkpoint),
        "baselines": task.baselines,
        "primary_metric": task.primary_metric,
        "variants": {
            v: {
                "n_params": r["n_params"],
                "best_val_loss": min(r["curves"]["val_loss"]),
                f"best_val_{task.primary_metric}": max(r["curves"][f"val_{task.primary_metric}"]),
                f"final_val_{task.primary_metric}": r["curves"][f"val_{task.primary_metric}"][-1],
                "curves": r["curves"],
            }
            for v, r in results.items()
        },
    }
    with (out_dir / "summary.json").open("w") as fp:
        json.dump({**vars(args), **summary}, fp, indent=2, default=str)

    pm = task.primary_metric
    print(f"\n=== {args.task} / {args.source} — best val {pm} ===")
    for v, r in summary["variants"].items():
        print(f"  {v:<14} {r[f'best_val_{pm}']:.3f}  "
              f"(final {r[f'final_val_{pm}']:.3f}, {r['n_params']:,} params)")
    refs = "  ".join(f"{k}={val:.3f}" for k, val in task.baselines.items())
    print(f"  baselines: {refs}")
    print(f"\nwrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
