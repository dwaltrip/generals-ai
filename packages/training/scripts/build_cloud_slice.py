#!/usr/bin/env -S uv run python
"""Build a self-contained slice of the intermediate corpus from a manifest.

Reads the replay IDs referenced by a manifest's `train` + `val` splits and
copies each game's `.npz` + `.meta.npz` into a slice directory, preserving the
`<rid[:2]>/` shard layout the parser uses. The manifest and its metrics
sidecar are copied alongside so the slice is a self-contained bundle ready to
upload to a Modal Volume (or anywhere else).

Example usage:
    build_cloud_slice.py \\
        --manifest data/training/splits/probe_500.json \\
        --out tmp/cloud-smoke-slice
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from settings import INTERMEDIATE_DIR
from training.analysis.run_metrics import get_manifest_metrics, sidecar_path
from training.bc.splits import load_manifest
from training.bc.utils import meta_path_for, sim_path_for
from utils.docstring import doc_summary


def build_slice(manifest_path: Path, intermediate_root: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to write into non-empty directory: {out_dir}")

    manifest = load_manifest(manifest_path)
    replay_ids = {rid for rid, _ in manifest["train"]} | {rid for rid, _ in manifest["val"]}

    out_intermediate = out_dir / "intermediate"
    out_intermediate.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    n_files = 0
    n_games = len(replay_ids)
    log_every = max(1, n_games // 10)  # ~10 progress lines per run, any size
    for i, rid in enumerate(sorted(replay_ids)):
        sim_src = sim_path_for(rid, intermediate_root)
        meta_src = meta_path_for(sim_src)
        sim_dst = sim_path_for(rid, out_intermediate)
        meta_dst = meta_path_for(sim_dst)
        sim_dst.parent.mkdir(parents=True, exist_ok=True)
        for src, dst in ((sim_src, sim_dst), (meta_src, meta_dst)):
            shutil.copy2(src, dst)
            total_bytes += dst.stat().st_size
            n_files += 1
        if (i + 1) % log_every == 0 or (i + 1) == n_games:
            print(f"  copied {i + 1:,} / {n_games:,} games  ({total_bytes / 1e6:.1f} MB)")

    manifest_dst = out_dir / manifest_path.name
    shutil.copy2(manifest_path, manifest_dst)

    # Ship the metrics sidecar with the manifest (backfilling it locally if
    # missing) so cloud-side consumers never need to walk the corpus.
    metrics = get_manifest_metrics(manifest_path, intermediate=intermediate_root, create=True)
    sidecar_dst = sidecar_path(manifest_dst)
    sidecar_dst.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"games:    {len(replay_ids)}")
    print(f"files:    {n_files} (sim + meta per game)")
    print(f"size:     {total_bytes / 1e6:.1f} MB")
    print(f"manifest: {manifest_dst}")
    print(f"sidecar:  {sidecar_dst}")
    print(f"slice:    {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=doc_summary(__doc__))
    parser.add_argument("--manifest", type=Path, required=True, help="path to manifest JSON")
    parser.add_argument(
        "--intermediate",
        type=Path,
        default=INTERMEDIATE_DIR,
        help=f"source intermediate corpus root (default: {INTERMEDIATE_DIR})",
    )
    parser.add_argument("--out", type=Path, required=True, help="slice output directory")
    args = parser.parse_args()
    build_slice(args.manifest.resolve(), args.intermediate.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
