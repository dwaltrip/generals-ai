"""The argmin-toy fq table spec.

One row per frame, carrying everything the toy needs: raw per-player `army_sim`
and `land_sim` (the encoder owns the `/1000`, so these stay in raw units), the
`alive` mask, and the analytic targets/diagnostics derived from them — `label`
(lowest-index argmin over alive), `margin` (gap to the second-lowest), `n_alive`,
and `argmin_set` (`[N,8]` membership in the tied set of true minima).

`label`, `margin`, and `argmin_set` all key off the one masked-army definition
(`_masked_army_totals`, `np.inf` sentinel for dead) so the tie boundary is shared
across the three accuracies (6.18-1 Risk #6). The masked army is the raw int64
sum, so the `==` in `argmin_set` is exact — normalization must not move upstream.
"""

from __future__ import annotations

import numpy as np

from training.analysis.families.army_derivers import ARMY_SIM, LAND_SIM
from training.analysis.families.elim_head_debug import (
    _masked_army_totals,
    bottom_two_margin,
    lowest_army_victim,
)
from training.analysis.fq.frame_table import FrameTable, FrameTableSpec


def n_alive(t: FrameTable) -> np.ndarray:
    """`[N]` count of alive players per frame."""
    return t.cols["alive"].sum(1)


def argmin_set(t: FrameTable) -> np.ndarray:
    """`[N, 8]` bool membership in the alive-argmin set: True for every alive
    player tied at the row's minimum army. Dead slots are `np.inf` (the shared
    masked-army convention) so they never equal the finite min → False. Credits a
    model that picks *a* true minimum even when it isn't the lowest-index one
    (`label`); the in-set accuracy (6.18-1 §6) reads this column."""
    masked = _masked_army_totals(t, np.inf)
    return masked == masked.min(1, keepdims=True)


ARGMIN_TOY = FrameTableSpec(
    name="argmin_toy",
    dataset_kwargs=dict(
        emit_alive_mask=True, include_frame_info=True, unsafe_attach_sim_frame=True
    ),
    emit_cols={"alive_mask": "alive"},
    derivers=[ARMY_SIM, LAND_SIM],
    derived_cols={
        "label": lowest_army_victim,
        "margin": bottom_two_margin,
        "n_alive": n_alive,
        "argmin_set": argmin_set,
    },
)
