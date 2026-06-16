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


ELIM_HEAD_DEBUG = FrameSpec(
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

# name -> FrameSpec, for the CLI's `--table` selection (a small registry of
# families, not of columns).
REGISTRY: dict[str, FrameSpec] = {ELIM_HEAD_DEBUG.name: ELIM_HEAD_DEBUG}
