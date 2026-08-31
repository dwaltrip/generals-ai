from __future__ import annotations

import numpy as np

from training.bc.datapipe.emit import emit_tail
from training.bc.datapipe.precompute import precompute_for
from training.bc.datapipe.sim_types import GameMeta, PerspectiveMeta
from training.goldens.registry import SupervisionEntry


def compute_supervision(
    sim: dict[str, np.ndarray],
    game_meta: GameMeta,
    persp: PerspectiveMeta,
    entry: SupervisionEntry,
) -> dict[str, np.ndarray]:
    T = persp.end_t
    assert T > 0, f"perspective has zero frames (end_t={T})"

    pre = precompute_for(entry.spec, sim)
    per_t = [
        emit_tail(sim, t, game_meta, persp, entry.spec, pre).to_dict()
        for t in range(T)
    ]
    # Every key is stacked per frame over t, including scalars like value_target.
    return {key: np.stack([d[key] for d in per_t]) for key in per_t[0]}
