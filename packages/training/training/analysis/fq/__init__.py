"""fq — frame-query analysis toolkit.

Walk a split once; derive named per-frame quantities via canonical single-source
derivers; collect them into a flat, dumb columnar `FrameTable` you slice with
numpy. See `docs/2026-06/6.15-3-fq-analysis-toolkit-design.md`.
"""

from __future__ import annotations

from training.analysis.fq.derivers import (
    ARMY_OBS,
    ARMY_SIM,
    Deriver,
    Frame,
    army_totals_per_tick,
    per_game,
)
from training.analysis.fq.families import REGISTRY, bottom_two_margin, lowest_army_victim
from training.analysis.fq.frame_table import (
    FrameSpec,
    FrameTable,
    build_frame_table,
    cap_by_games,
    check_representative,
    select,
)


__all__ = [
    "ARMY_OBS",
    "ARMY_SIM",
    "Deriver",
    "Frame",
    "FrameSpec",
    "FrameTable",
    "REGISTRY",
    "army_totals_per_tick",
    "bottom_two_margin",
    "build_frame_table",
    "cap_by_games",
    "check_representative",
    "lowest_army_victim",
    "per_game",
    "select",
]
