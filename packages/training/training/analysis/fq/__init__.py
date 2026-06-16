"""fq — frame-query analysis toolkit.

Walk a split once; derive named per-frame quantities via canonical single-source
derivers; collect them into a flat, dumb columnar `FrameTable` you slice with
numpy. See `docs/2026-06/6.15-3-fq-analysis-toolkit-design.md`.
"""

from __future__ import annotations

from training.analysis.fq.derivers import (
    Deriver,
    Frame,
    per_game,
)
from training.analysis.fq.registry import REGISTRY
from training.analysis.fq.frame_table import (
    FrameSpec,
    FrameTable,
    build_frame_table,
    cap_by_games,
    check_representative,
    select,
)


__all__ = [
    "Deriver",
    "Frame",
    "FrameSpec",
    "FrameTable",
    "REGISTRY",
    "build_frame_table",
    "cap_by_games",
    "check_representative",
    "per_game",
    "select",
]
