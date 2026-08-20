"""The argmin-toy fq table spec.

One row per frame, carrying the target-independent quantities: raw per-player
`army_sim` and `land_sim` (the encoder owns the `/1000`, so these stay in raw
units), the `alive` mask, the `captured` flag (prod's capture-based status, §6),
`n_alive`, and `surr_frame` (the surrender-subpopulation selector).

The target-dependent trio — `label` (argmin), `margin` (gap to the second-lowest),
and `argmin_set` (membership in the tied minima) — is *not* built here: it keys off
the target's membership mask (`alive` vs `army>0`), so `data.prep_target` computes
it per target off the one masked-army definition, keeping the tie boundary consistent
across the three (6.18-4 §8 #2).
"""

from __future__ import annotations

import numpy as np

from training.analysis.families.army_derivers import ARMY_SIM, CAPTURED, LAND_SIM
from training.analysis.fq.frame_table import FrameTable, FrameTableSpec
from training.bc.config.targets_config import TargetsConfig
from training.bc.emit_spec import PartialEmitSpec


def n_alive(t: FrameTable) -> np.ndarray:
    """`[N]` count of alive players per frame."""
    return t.cols["alive"].sum(1)


def surr_frame(t: FrameTable) -> np.ndarray:
    """`[N]` bool: frame holds a surrendered-present slot (`~alive & army>0`). The
    surrender-subpopulation selector for the metric slice (6.18-4 §8 #1); a data
    property, independent of the target."""
    return np.asarray(((~t.cols["alive"].astype(bool)) & (t.cols["army_sim"] > 0)).any(1))


ARGMIN_TOY = FrameTableSpec(
    name="argmin_toy",
    emit=PartialEmitSpec(
        targets=TargetsConfig(elim_variant=None, elim_bin_edges=None),
        emit_alive_mask=True,
        attach_sim_frame=True,
    ),
    emit_cols={"alive_mask": "alive"},
    derivers=[ARMY_SIM, LAND_SIM, CAPTURED],
    derived_cols={"n_alive": n_alive, "surr_frame": surr_frame},
)
