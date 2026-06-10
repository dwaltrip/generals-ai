"""Marginal placement statistics for a BC split manifest — pure compute.

The marginal-entropy floor: the val value loss of a model that ignores the
board and predicts the split's placement frequencies. Computed two ways:

  - frame-weighted: each (perspective, game) contributes `n_frames` samples,
    matching how the value loss is averaged during training (mean-over-batch).
    This is the apples-to-apples floor for `val_value` in `epochs.jsonl`.
  - perspective-weighted: each (perspective, game) contributes 1 sample.

Sidecar persistence and run-dir resolution live in `run_metrics`; the CLI is
`scripts/marginal_entropy.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from training.bc.splits import load_manifest, samples_for_split


N_CLASSES = 8  # FFA placement: 1..8 → class index 0..7

H_UNIFORM = math.log(N_CLASSES)


@dataclass(frozen=True)
class SplitMarginals:
    """Placement distribution of one manifest split, both weightings."""

    split: str
    n_perspectives: int
    n_frames: int
    frame_counts: tuple[int, ...]        # per placement class, frame-weighted
    perspective_counts: tuple[int, ...]  # per placement class, one per perspective

    @property
    def h_frame(self) -> float:
        """Frame-weighted marginal entropy (nats) — the `val_value` floor."""
        return entropy_nats(np.asarray(self.frame_counts))

    @property
    def h_persp(self) -> float:
        return entropy_nats(np.asarray(self.perspective_counts))


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


def split_marginals(
    manifest_path: Path, intermediate: Path, split: str = "val"
) -> SplitMarginals:
    """Compute a split's placement marginals by walking its perspectives'
    `meta.npz` files (placement, elim timestep) and sim files (T)."""
    manifest = load_manifest(manifest_path)
    samples = samples_for_split(manifest, split, intermediate)

    frame_counts = np.zeros(N_CLASSES, dtype=np.int64)
    persp_counts = np.zeros(N_CLASSES, dtype=np.int64)
    sim_T_cache: dict[Path, int] = {}

    for sim_path, k in samples:
        n_frames, placement = n_frames_for(sim_path, k, sim_T_cache)
        frame_counts[placement] += n_frames
        persp_counts[placement] += 1

    return SplitMarginals(
        split=split,
        n_perspectives=len(samples),
        n_frames=int(frame_counts.sum()),
        frame_counts=tuple(int(c) for c in frame_counts),
        perspective_counts=tuple(int(c) for c in persp_counts),
    )
