"""Run-metrics collation: the IO seam between run dirs / manifests and the
display builders.

Owns the manifest metrics sidecar (`<stem>.metrics.json`, placement marginals
today — new metric keys slot in beside `val_marginals` over time) and the
run-dir → manifest resolution needed to read it for any historical run. The
division of labor: display code (`run_comparison`, report renderers) stays a
pure function of values; runners — CLIs, report generation, manifest creation
— call in here, passing `create=True` where backfilling a missing sidecar is
appropriate. This module is the intended home for future run-metric
normalization (legacy args schemas, checkpoint-era architecture lookup).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from settings import INTERMEDIATE_DIR
from training.analysis.marginal_entropy import SplitMarginals, split_marginals
from training.settings import SPLITS_DIR


def sidecar_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(manifest_path.stem + ".metrics.json")


def _manifest_hash(manifest_path: Path) -> str:
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _marginals_dict(m: SplitMarginals) -> dict:
    return {
        "n_perspectives": m.n_perspectives,
        "n_frames": m.n_frames,
        "frame_counts": list(m.frame_counts),
        "perspective_counts": list(m.perspective_counts),
        "H_frame": round(m.h_frame, 6),
        "H_persp": round(m.h_persp, 6),
    }


def compute_manifest_metrics(
    manifest_path: Path, intermediate: Path = INTERMEDIATE_DIR
) -> dict:
    """The sidecar content for one manifest: val placement marginals (the
    `val_value` floor is `val_marginals.H_frame`). Walks the corpus, so it
    needs `intermediate` present — ~1s per 1k val perspectives."""
    return {
        "manifest": manifest_path.name,
        "manifest_sha256": _manifest_hash(manifest_path),
        "val_marginals": _marginals_dict(
            split_marginals(manifest_path, intermediate, "val")
        ),
    }


def write_manifest_metrics(
    manifest_path: Path, intermediate: Path = INTERMEDIATE_DIR
) -> dict:
    """Compute the sidecar and write it next to the manifest. The
    manifest-creation hook (`bc.splits` build, `subset_manifest.py`)."""
    content = compute_manifest_metrics(manifest_path, intermediate)
    sidecar_path(manifest_path).write_text(json.dumps(content, indent=2) + "\n")
    return content


def get_manifest_metrics(
    manifest_path: Path,
    *,
    intermediate: Path = INTERMEDIATE_DIR,
    create: bool = False,
) -> dict | None:
    """Load a manifest's metrics sidecar; a sidecar whose `manifest_sha256`
    doesn't match the manifest's current content counts as missing (manifests
    are write-once, so a mismatch means a `--force` rebuild left the sidecar
    behind). With `create`, a missing sidecar is computed and backfilled
    (write is best-effort — a read-only location still returns the computed
    values); without it, missing returns None.
    """
    sc = sidecar_path(manifest_path)
    if sc.is_file():
        try:
            content = json.loads(sc.read_text())
            if content.get("manifest_sha256") == _manifest_hash(manifest_path):
                return content
        except (OSError, json.JSONDecodeError):
            pass
    if not create:
        return None
    content = compute_manifest_metrics(manifest_path, intermediate)
    try:
        sc.write_text(json.dumps(content, indent=2) + "\n")
    except OSError:
        pass
    return content


def resolve_manifest(raw: str | Path, splits_dir: Path = SPLITS_DIR) -> Path | None:
    """Resolve a manifest path as recorded in a run's `args.json`.

    Historical runs record paths that no longer exist locally — relative paths
    from before the data-layout moves, or cloud-container absolute paths
    (`/data/cloud_24k.json`). Use the path when it exists; otherwise fall back
    to a basename lookup in the local splits dir. The fallback assumes
    same-name = same-manifest, which holds while manifests stay write-once
    (both creation CLIs refuse to overwrite without --force).
    """
    p = Path(raw)
    if p.is_file():
        return p
    candidate = splits_dir / p.name
    if candidate.is_file():
        return candidate
    return None


def floor_for_run(run_dir: Path, *, create: bool = False) -> float | None:
    """The frame-weighted val marginal entropy of the manifest a run trained
    on — the floor to read `val_value` against. None when the run's manifest
    can't be resolved (or has no sidecar and `create` is off)."""
    try:
        args = json.loads((run_dir / "args.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    raw = args.get("manifest")
    if not raw:
        return None
    manifest = resolve_manifest(raw)
    if manifest is None:
        return None
    # Prefer the run's own corpus root (e.g. the slice on a cloud host) over
    # the local default when a compute is needed.
    recorded = args.get("intermediate")
    intermediate = (
        Path(recorded) if recorded and Path(recorded).is_dir() else INTERMEDIATE_DIR
    )
    metrics = get_manifest_metrics(manifest, intermediate=intermediate, create=create)
    if metrics is None:
        return None
    return (metrics.get("val_marginals") or {}).get("H_frame")
