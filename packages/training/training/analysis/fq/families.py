"""Curated families (bundled emit + columns) and their derived columns.

Families are hand-picked bundles, not a global registry of every column. The
derived columns here are plain numpy over a built table — including the
lowest-army *rule* as a computed column, so any head-vs-rule comparison is
matched to the eval frames by construction (the 6.14-1 hardcoded-baseline fix).
"""

from __future__ import annotations

import numpy as np

from training.analysis.fq.derivers import ARMY_OBS, ARMY_SIM
from training.analysis.fq.frame_table import FrameSpec, FrameTable


def lowest_army_victim(t: FrameTable) -> np.ndarray:
    """`[N]` channel of the lowest-army alive player — the lowest-army who-dies-next
    rule, evaluated on the table's exact frames."""
    masked = np.where(t.cols["alive"], t.cols["army_sim"], np.inf)
    return masked.argmin(1)


def bottom_two_margin(t: FrameTable) -> np.ndarray:
    """`[N]` army gap between the two lowest-army alive players; `-1` when <2 are
    alive. Small margins are where the lowest-army rule (and the head) struggle."""
    s = np.sort(np.where(t.cols["alive"], t.cols["army_sim"], np.inf), axis=1)
    m = s[:, 1] - s[:, 0]
    m[~np.isfinite(m)] = -1.0
    return m


ELIM_HEAD_DEBUG = FrameSpec(
    name="elim_head_debug",
    dataset_kwargs=dict(
        elim_head_variant="next_death", include_frame_info=True, unsafe_attach_sim_frame=True
    ),
    emit_cols={"alive_mask": "alive", "next_elim_target": "victim", "next_elim_dt": "dt"},
    derivers=[ARMY_OBS, ARMY_SIM],
    derived_cols={"margin": bottom_two_margin, "low_army_victim": lowest_army_victim},
)

# name -> FrameSpec, for the CLI's `--table` selection (a small registry of
# families, not of columns).
REGISTRY: dict[str, FrameSpec] = {ELIM_HEAD_DEBUG.name: ELIM_HEAD_DEBUG}
