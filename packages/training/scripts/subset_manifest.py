#!/usr/bin/env -S uv run python
"""Build a smaller manifest as a subset of an existing one.

Takes a source manifest and writes a derived manifest with a prefix-slice
of its train + val pair lists (val scaled proportionally to source's
`val_frac`). Useful for sizing throughput benchmarks against a slice that
contains all the source's games — any subset's rids are a subset of the
source's, so they all resolve under the same intermediate corpus.

The source manifest's `train` and `val` lists are seeded-shuffled by
`bc.splits.build_manifest`, so a prefix is itself a uniform random sample.
No re-shuffle or extra seed needed.

Output preserves all source fields (seed, filter_version, git_sha, etc.)
except `train`, `val`, and `kept_pairs` — those fields still describe the
upstream build, which is still true of any prefix-subset.

Example usage:
    subset_manifest.py \\
        --in data/training/splits/cloud_24k.json \\
        --max-pairs 5000 \\
        --out data/training/splits/cloud_5k.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from training.bc.splits import load_manifest
from utils.docstring import doc_summary
from utils.json_io import write_json


def subset_manifest(src: dict, max_pairs: int) -> dict:
    src_total = len(src["train"]) + len(src["val"])
    if max_pairs > src_total:
        raise SystemExit(f"--max-pairs {max_pairs} exceeds source total {src_total}")
    val_frac = src["val_frac"]
    n_val = max(1, round(max_pairs * val_frac))
    n_train = max_pairs - n_val
    if n_train < 1:
        raise SystemExit(
            f"--max-pairs {max_pairs} too small after val carve-out (n_val={n_val})"
        )

    out = dict(src)
    out["train"] = src["train"][:n_train]
    out["val"] = src["val"][:n_val]
    out["kept_pairs"] = n_train + n_val
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=doc_summary(__doc__))
    parser.add_argument(
        "--in",
        dest="src",
        type=Path,
        required=True,
        help="path to source manifest",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        required=True,
        help="total train+val pairs to keep (val scaled proportionally to source val_frac)",
    )
    parser.add_argument("--out", type=Path, required=True, help="path to write derived manifest")
    parser.add_argument("--force", action="store_true", help="overwrite existing out path")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite: {args.out} (pass --force)")

    src = load_manifest(args.src)
    derived = subset_manifest(src, args.max_pairs)
    write_json(args.out, derived)

    print(f"source:  {args.src} ({len(src['train']):,} train + {len(src['val']):,} val)")
    print(f"derived: {args.out} ({len(derived['train']):,} train + {len(derived['val']):,} val)")


if __name__ == "__main__":
    main()
