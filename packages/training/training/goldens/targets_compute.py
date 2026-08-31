from __future__ import annotations

import numpy as np

from training.bc.datapipe.emit import emit_frame
from training.bc.datapipe.precompute import precompute_for
from training.bc.datapipe.sim_types import GameMeta, PerspectiveMeta
from training.goldens.registry import TargetsEntry


def compute_targets(
    sim: dict[str, np.ndarray],
    game_meta: GameMeta,
    persp: PerspectiveMeta,
    entry: TargetsEntry,
) -> dict[str, np.ndarray]:
    T = persp.end_t
    assert T > 0, f"perspective has zero frames (end_t={T})"

    pre = precompute_for(entry.spec, sim)
    emitted_per_t = [
        emit_frame(sim, t, game_meta, persp, entry.spec, pre).to_dict()
        for t in range(T)
    ]
    # Every key is stacked per frame over t, including scalars like value_target.
    return {
        key: np.stack([emission_dict[key] for emission_dict in emitted_per_t])
        for key in emitted_per_t[0]
    }
