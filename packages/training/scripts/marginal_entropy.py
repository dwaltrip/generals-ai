#!/usr/bin/env -S uv run python
"""Marginal placement entropy for a BC split manifest.

Prints the entropy of the val (and optionally train) placement distribution
two ways:

  - frame-weighted: each (perspective, game) contributes `n_frames` samples,
    matching how the value loss is averaged during training (mean-over-batch).
  - perspective-weighted: each (perspective, game) contributes 1 sample.

The frame-weighted entropy is the apples-to-apples floor to compare against
the val value loss in `epochs.jsonl`. A model that has converged but ignores
the board ("input-agnostic, learned-the-marginal") would bottom out here.

`--write` persists the val marginals as the manifest's `<stem>.metrics.json`
sidecar (normally created with the manifest; this backfills older ones).

Example usage:
    marginal_entropy.py data/training/splits/poc_2kish_perspecs.json
    marginal_entropy.py <manifest> --intermediate <path>
    marginal_entropy.py <manifest> --write

This is a CLI-wrapper around `training.analysis.marginal_entropy`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from settings import INTERMEDIATE_DIR
from training.analysis.marginal_entropy import (
    H_UNIFORM,
    N_CLASSES,
    SplitMarginals,
    split_marginals,
)
from training.analysis.run_metrics import sidecar_path, write_manifest_metrics
from utils.docstring import doc_summary


def summarize(m: SplitMarginals) -> None:
    def fmt_dist(counts: tuple[int, ...]) -> str:
        total = sum(counts)
        if total == 0:
            return "(empty)"
        return "  ".join(f"{i + 1}={counts[i] / total:.3f}" for i in range(N_CLASSES))

    print(f"\n=== {m.split} ===")
    print(f"  {m.n_perspectives:,} perspectives, {m.n_frames:,} frames")
    print(f"  frame-weighted   p:  {fmt_dist(m.frame_counts)}")
    print(f"  perspective-wt   p:  {fmt_dist(m.perspective_counts)}")
    print(f"  H(frame-weighted)        = {m.h_frame:.4f} nats")
    print(f"  H(perspective-weighted)  = {m.h_persp:.4f} nats")
    print(f"  H(uniform reference)     = {H_UNIFORM:.4f} nats  (= ln {N_CLASSES})")


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
    p.add_argument("--write", action="store_true",
                   help="Write the val marginals to the manifest's metrics sidecar")
    args = p.parse_args()

    intermediate = args.intermediate.resolve()

    summarize(split_marginals(args.manifest, intermediate, "val"))

    if args.include_train:
        summarize(split_marginals(args.manifest, intermediate, "train"))

    if args.write:
        write_manifest_metrics(args.manifest, intermediate)
        print(f"\nwrote {sidecar_path(args.manifest)}")


if __name__ == "__main__":
    main()
