"""Marginal placement entropy for a BC split manifest.

Computes the entropy of the val (and optionally train) placement distribution
two ways:

  - frame-weighted: each (perspective, game) contributes `n_frames` samples,
    matching how the value loss is averaged during training (mean-over-batch).
  - perspective-weighted: each (perspective, game) contributes 1 sample.

The frame-weighted entropy is the apples-to-apples floor to compare against
the val value loss in `epochs.jsonl`. A model that has converged but ignores
the board ("input-agnostic, learned-the-marginal") would bottom out here.

Example usage:
    marginal_entropy.py data/training/splits/poc_2kish_perspecs.json
    marginal_entropy.py <manifest> --intermediate <path>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from settings import INTERMEDIATE_DIR
from training.bc.splits import load_manifest, samples_for_split
from utils.docstring import doc_summary


N_CLASSES = 8  # FFA placement: 1..8 → class index 0..7


def n_frames_for(sim_path: Path, k: int, sim_T_cache: dict[Path, int]) -> tuple[int, int]:
    """Return (n_frames_contributed, placement_0_indexed) for one val pair."""
    meta_path = sim_path.with_name(sim_path.stem + ".meta.npz")
    with np.load(meta_path) as meta:
        placement = int(meta["placement"][k]) - 1
        elim_t = int(meta["elim_timestep"][k])

    if sim_path not in sim_T_cache:
        with np.load(sim_path) as sim:
            sim_T_cache[sim_path] = int(sim["ownership"].shape[0])
    T = sim_T_cache[sim_path]

    end_t = T - 1 if elim_t == -1 else min(T - 1, elim_t)
    return end_t, placement


def entropy_nats(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum())


def summarize(name: str, samples: list[tuple[Path, int]]) -> None:
    frame_counts = np.zeros(N_CLASSES, dtype=np.int64)
    persp_counts = np.zeros(N_CLASSES, dtype=np.int64)
    sim_T_cache: dict[Path, int] = {}

    for sim_path, k in samples:
        n_frames, placement = n_frames_for(sim_path, k, sim_T_cache)
        frame_counts[placement] += n_frames
        persp_counts[placement] += 1

    def fmt_dist(counts: np.ndarray) -> str:
        total = counts.sum()
        if total == 0:
            return "(empty)"
        return "  ".join(f"{i + 1}={counts[i] / total:.3f}" for i in range(N_CLASSES))

    H_frame = entropy_nats(frame_counts)
    H_persp = entropy_nats(persp_counts)
    H_uniform = math.log(N_CLASSES)

    print(f"\n=== {name} ===")
    print(f"  {len(samples):,} perspectives, {frame_counts.sum():,} frames")
    print(f"  frame-weighted   p:  {fmt_dist(frame_counts)}")
    print(f"  perspective-wt   p:  {fmt_dist(persp_counts)}")
    print(f"  H(frame-weighted)        = {H_frame:.4f} nats")
    print(f"  H(perspective-weighted)  = {H_persp:.4f} nats")
    print(f"  H(uniform reference)     = {H_uniform:.4f} nats  (= ln 8)")


def main() -> None:
    p = argparse.ArgumentParser(description=doc_summary(__doc__))
    p.add_argument("manifest", type=Path)
    p.add_argument(
        "--intermediate", type=Path,
        default=INTERMEDIATE_DIR,
        help=f"Path to intermediate corpus root (default: {INTERMEDIATE_DIR})",
    )
    p.add_argument("--include-train", action="store_true",
                   help="Also compute train marginal (slower; 95%% of corpus)")
    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    intermediate = args.intermediate.resolve()

    val_samples = samples_for_split(manifest, "val", intermediate)
    summarize("val", val_samples)

    if args.include_train:
        train_samples = samples_for_split(manifest, "train", intermediate)
        summarize("train", train_samples)


if __name__ == "__main__":
    main()
