"""The elim_head_debug family: the table for debugging the who-dies-next elim head.

Pairs the head's next_death targets with sim/obs army and the lowest-army rule as
derived columns, so head-vs-rule comparisons are matched to the eval frames by
construction. Join a model dump (`join_dump`) to add the head's predictions.

NOTE (follow-up): the next_death head's domain is now board-*removal* (the
`present` field), but this family still keys the softmax/rule domain on
`alive_mask` (`emit_cols` + the `_masked_army_totals` rule). The two differ only
on the surrender window (~1% of frames), so the head-vs-rule reads are close but
not exact. Aligning this to `present_mask` (the dataset emits it) is deferred —
it ripples to the `low_army_victim` rule, the `join_dump` truth_map, and the
`head_vs_rule_by_margin` consumer, so it wants its own pass.
"""

from __future__ import annotations

import numpy as np

from training.analysis.families.army_derivers import (
    ARMY_OBS,
    ARMY_SIM,
)
from training.analysis.fq.frame_table import FrameTable, FrameTableSpec
from training.bc.config.targets_config import TARGETS_CFG_ELIM_NEXT_DEATH
from training.bc.emit_spec import PartialEmitSpec


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
    emit=PartialEmitSpec(
        targets=TARGETS_CFG_ELIM_NEXT_DEATH,
        emit_alive_mask=True,
        attach_sim_frame=True,
    ),
    emit_cols={
        "alive_mask": "alive",
        "next_elim_target": "victim",
        "next_elim_dt": "dt",
        # Board-removal/present domain (the head's actual domain): the per-frame
        # present mask and the per-channel removal horizon that feeds the soft
        # target (`p_i ∝ exp(-removal_dt_i/τ)`). Additive — for soft-target shape
        # diagnostics; leaves the alive/rule columns above untouched.
        "present_mask": "present",
        "next_elim_removal_dt": "removal_dt",
    },
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
