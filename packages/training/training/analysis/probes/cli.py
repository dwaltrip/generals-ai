"""Reusable CLI scaffolding for frozen-representation probes.

A probe run is a thin script: build the task, call `run_probe_cli`. The arg
parser and the load-checkpoint → build-source → run → summarize flow live here
so one-off probe scripts (`training/analysis/probe_runs/`) stay minimal — they
contribute only their `ProbeTask` and a 3-line `main`.

The trunk checkpoint is authoritative for the obs config: `--source raw_obs`
loads it only to read that config, so the control reads the exact obs tensor the
trunk was trained on. The task is built from the resolved `ObsConfig` (a factory)
so tasks that need it — e.g. army-channel indices — get it.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import io
import json
from pathlib import Path
import sys
from typing import Callable

import torch

from settings import INTERMEDIATE_DIR
from training.analysis.probes.core import (
    ProbeTask,
    RawObsSource,
    TrunkSource,
    load_frozen_model,
    run_probe,
)
from training.bc.obs_config import ObsConfig
from training.bc.splits import load_manifest, samples_for_split
from training.settings import PROBE_OUTPUT_DIR
from training.shared.device import disable_mps_fallback, pick_device


class _Tee:
    """Mirror stdout writes to the console and, once attached, a log file. Output
    before `attach` is buffered so the log captures the run from its first line —
    the run dir's name needs the task, which isn't known until the model loads."""

    def __init__(self, console):
        self._console = console
        self._buffer = io.StringIO()
        self._file: io.TextIOWrapper | None = None

    def write(self, s: str) -> int:
        self._console.write(s)
        (self._file or self._buffer).write(s)
        return len(s)

    def flush(self) -> None:
        self._console.flush()
        (self._file or self._buffer).flush()

    def attach(self, path: Path) -> None:
        self._file = open(path, "w")
        self._file.write(self._buffer.getvalue())  # replay the buffered prefix

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def build_probe_arg_parser(description: str) -> argparse.ArgumentParser:
    """The shared probe arg surface. One-off scripts may add their own args."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--source", choices=("trunk", "raw_obs"), default="trunk")
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Trunk checkpoint. For --source raw_obs, used only to read the obs "
        "config so the control matches the trunk run's inputs.",
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
        help="Cap train_probe frames (smoke / memory). Default: all.",
    )
    p.add_argument("--max-val-frames", type=int, default=None)
    p.add_argument("--out-root", type=Path, default=PROBE_OUTPUT_DIR)
    return p


def run_probe_cli(
    task_factory: Callable[[ObsConfig], ProbeTask], args: argparse.Namespace
) -> dict:
    """Load the frozen checkpoint, build the feature source + task, run the probe,
    write `summary.json`, and print a per-variant table of every metric the task
    reports against its baselines."""
    disable_mps_fallback()
    for path in (args.checkpoint, args.manifest, args.intermediate):
        if not path.exists():
            raise SystemExit(f"not found: {path}")

    # Tee all stdout to the run dir's run.log (created mid-run, once the task name
    # is known); buffered until then so the log captures the run from line one.
    tee = _Tee(sys.stdout)
    sys.stdout = tee
    try:
        device = pick_device(args.device)
        torch.manual_seed(args.seed)
        print(f"device: {device}")

        model = load_frozen_model(args.checkpoint, device)
        obs_cfg = model.cfg.obs
        print(f"obs config: dense_history_n={obs_cfg.dense_history_n} "
              f"({obs_cfg.obs_channels} channels)")

        if args.source == "trunk":
            source = TrunkSource(model.trunk, out_channels=model.cfg.outer_width)
        else:
            source = RawObsSource(out_channels=obs_cfg.obs_channels)

        task = task_factory(obs_cfg)
        manifest = load_manifest(args.manifest)
        val_samples = samples_for_split(manifest, "val", args.intermediate)
        print(f"manifest: {args.manifest.name}  ({len(val_samples)} val perspectives)")

        out_dir = args.out_root / f"{datetime.now():%Y%m%d-%H%M%S}-{task.name}-{args.source}"
        out_dir.mkdir(parents=True, exist_ok=False)
        tee.attach(out_dir / "run.log")
        print(f"out dir: {out_dir}\n")

        results = run_probe(
            task, source, val_samples, obs_cfg,
            device=device, seed=args.seed, val_frac=args.val_frac,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            cache_chunk=args.cache_chunk,
            max_train_frames=args.max_train_frames, max_val_frames=args.max_val_frames,
        )

        # Every "val_*" curve except the loss is a task metric — report best + final.
        sample_curves = next(iter(results.values()))["curves"]
        metric_keys = [k[len("val_"):] for k in sample_curves
                       if k.startswith("val_") and k != "val_loss"]
        variants = {}
        for v, r in results.items():
            c = r["curves"]
            entry = {"n_params": r["n_params"], "best_val_loss": min(c["val_loss"])}
            for mk in metric_keys:
                entry[f"best_val_{mk}"] = max(c[f"val_{mk}"])
                entry[f"final_val_{mk}"] = c[f"val_{mk}"][-1]
            entry["curves"] = c
            variants[v] = entry

        summary = {
            "task": task.name, "source": args.source, "checkpoint": str(args.checkpoint),
            "baselines": task.baselines, "primary_metric": task.primary_metric,
            "variants": variants,
        }
        with (out_dir / "summary.json").open("w") as fp:
            json.dump({**vars(args), **summary}, fp, indent=2, default=str)

        print(f"\n=== {task.name} / {args.source} — best val ===")
        for v, e in variants.items():
            cols = "  ".join(f"{mk} {e[f'best_val_{mk}']:.3f}" for mk in metric_keys)
            print(f"  {v:<14} {cols}  ({e['n_params']:,} params)")
        refs = "  ".join(f"{k}={val:.3f}" for k, val in task.baselines.items())
        print(f"  baselines: {refs}")
        print(f"\nwrote: {out_dir / 'summary.json'}")
        return results
    finally:
        sys.stdout = tee._console
        tee.close()
