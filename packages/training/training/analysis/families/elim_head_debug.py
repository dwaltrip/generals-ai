"""The elim_head_debug family: the table for debugging the who-dies-next elim head.

Pairs the head's next_death targets with sim/obs army and the lowest-army rule as
derived columns, so head-vs-rule comparisons are matched to the eval frames by
construction. Join a model dump (`join_dump`) to add the head's predictions.
"""

from __future__ import annotations

import numpy as np

from training.analysis.families.army_derivers import (
    ARMY_OBS,
    ARMY_SIM,
)
from training.analysis.fq.frame_table import FrameTable, FrameTableSpec


def lowest_army_victim(t: FrameTable) -> np.ndarray:
    """`[N]` per-frame channel of player with smallest total army.
    This is the "lowest-army" heuristic for the who-dies-next elim-head."""
    return _masked_army_totals(t, np.inf).argmin(1)


def bottom_two_margin(t: FrameTable) -> np.ndarray:
    """`[N]` army gap between the two lowest-army alive players; `-1` when <2 are
    alive. Small margins are where the lowest-army rule (and the head) struggle."""
    masked_sorted = np.sort(_masked_army_totals(t, np.inf), axis=1)
    gap = masked_sorted[:, 1] - masked_sorted[:, 0]
    gap[~np.isfinite(gap)] = -1.0
    return gap


def _masked_army_totals(t: FrameTable, sentinel: float) -> np.ndarray:
    return np.where(t.cols["alive"], t.cols["army_sim"], sentinel)


ELIM_HEAD_DEBUG = FrameTableSpec(
    name="elim_head_debug",
    dataset_kwargs=dict(
        elim_head_variant="next_death", include_frame_info=True, unsafe_attach_sim_frame=True
    ),
    emit_cols={"alive_mask": "alive", "next_elim_target": "victim", "next_elim_dt": "dt"},
    derivers=[ARMY_OBS, ARMY_SIM],
    derived_cols={"margin": bottom_two_margin, "low_army_victim": lowest_army_victim},
    # Shared ground truth for join_dump: these table cols must equal the dump's
    # next_elim_* cols on the join overlap (the persp_val_index cross-check).
    truth_map={
        "victim": "next_elim_target",
        "dt": "next_elim_dt",
        "alive": "next_elim_alive_mask",
    },
)
