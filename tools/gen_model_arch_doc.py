#!/usr/bin/env -S uv run python
"""gen_model_arch_doc — render the model-architecture doc and its diagrams.

Run from repo root:
    ./tools/gen_model_arch_doc.py            # regenerate docs/ in place
    ./tools/gen_model_arch_doc.py --check    # fail if docs/ is stale (no writes)
    ./tools/gen_model_arch_doc.py --png      # also rasterize to tmp/ for review

The doc + SVGs are derived from MODEL_CONFIG_DEFAULTS, so `--check` (a
byte-comparison of freshly generated content against the committed files)
guards against the artifacts drifting from the model code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import drawsvg as dw
from model_arch.diagrams import overview_diagram, u_diagram
from model_arch.doc import build_doc
from model_arch.spec import overview_spec, trunk_spec

from training.bc.model_config import MODEL_CONFIG_DEFAULTS


REPO_ROOT = Path(__file__).resolve().parent.parent
DIAGRAMS_DIR = REPO_ROOT / "docs" / "diagrams"
DOC_PATH = REPO_ROOT / "docs" / "model-architecture-overview.md"


def build_diagrams() -> dict[str, dw.Drawing]:
    """name → drawsvg.Drawing for every figure the doc embeds."""
    cfg = MODEL_CONFIG_DEFAULTS
    trunk = trunk_spec(cfg)
    ov = overview_spec(cfg)
    return {
        "model-overview": overview_diagram(
            ov.obs, ov.trunk, ov.embedding, ov.heads,
        ),
        "trunk-pyramid-u": u_diagram(
            trunk.granular, in_label=trunk.in_label, out_label=trunk.out_label,
            skip_note=f"skips: encoder block input → mirror decoder block "
                      f"({trunk.n_skips} total)",
        ),
        "trunk-pyramid-u-compact": u_diagram(
            trunk.compact, in_label=trunk.in_label, out_label=trunk.out_label,
            skip_note=f"skips: per-block, encoder input → mirror decoder "
                      f"({trunk.n_skips} total, bundled per level)",
        ),
    }


def build_artifacts() -> dict[Path, str]:
    """Canonical output path → file content, for every generated artifact.

    The single source of truth shared by the write and `--check` paths, so
    the two can never disagree about what *should* be on disk.
    """
    raw_artifacts = {
        (DIAGRAMS_DIR / f"{name}.svg"): drawing.as_svg()
        for name, drawing in build_diagrams().items()
    }
    raw_artifacts[DOC_PATH] = build_doc(MODEL_CONFIG_DEFAULTS)
    # drawsvg's as_svg() is typed str | None; drop any that came back empty so
    # a partial render can't silently overwrite a good committed artifact.
    artifacts = {}
    for path, content in raw_artifacts.items():
        if content is None:
            print(f"WARNING: could not render {path.name}, skipping.")
        else:
            artifacts[path] = content
    return artifacts


def write(png: bool) -> None:
    artifacts = build_artifacts()
    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path}")
    if png:
        _rasterize(p for p in artifacts if p.suffix == ".svg")


def check() -> int:
    """Compare committed artifacts against freshly generated content. Returns a
    process exit code (0 = up to date, 1 = stale)."""
    stale = [path for path, content in build_artifacts().items()
             if not path.exists() or path.read_text(encoding="utf-8") != content]
    if stale:
        print("model-architecture artifacts are stale:")
        for path in stale:
            print(f"  {path.relative_to(REPO_ROOT)}")
        print("regenerate with: ./tools/gen_model_arch_doc.py")
        return 1
    print("model-architecture artifacts up to date")
    return 0


def _rasterize(svg_paths) -> None:
    """Rasterize SVGs to tmp/ via rsvg-convert, for visual review."""
    if not shutil.which("rsvg-convert"):
        print("  (rsvg-convert not found; skipping PNG)")
        return
    (REPO_ROOT / "tmp").mkdir(exist_ok=True)
    for svg in svg_paths:
        png = REPO_ROOT / "tmp" / f"{svg.stem}.png"
        subprocess.run(["rsvg-convert", "-z", "2", "-b", "white",
                        str(svg), "-o", str(png)], check=True)
        print(f"wrote {png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed docs/ artifacts are stale")
    ap.add_argument("--png", action="store_true",
                    help="also rasterize each diagram to tmp/ for review")
    args = ap.parse_args()

    if args.check:
        raise SystemExit(check())
    write(png=args.png)


if __name__ == "__main__":
    main()
