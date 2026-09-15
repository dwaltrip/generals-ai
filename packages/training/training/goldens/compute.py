from __future__ import annotations

import numpy as np

from training.bc.datapipe.emit import emit_tail
from training.bc.datapipe.emit_spec import PartialEmitSpec
from training.bc.datapipe.precompute import precompute_for
from training.bc.datapipe.sim_types import PerspectiveMeta, SimGame
from training.bc.datapipe.walk import walk
from training.bc.obs_config import ObsConfig
from training.goldens.hashes import ObsDigest, ObsHasher


def compute_obs(game: SimGame, persp: PerspectiveMeta, cfg: ObsConfig) -> ObsDigest:
    hasher = ObsHasher()
    for frame in walk(game, persp, cfg):
        hasher.add(frame.obs)
    return hasher.digest()


def compute_supervision(
    game: SimGame, persp: PerspectiveMeta, spec: PartialEmitSpec
) -> dict[str, np.ndarray]:
    T = persp.end_t
    assert T > 0, f"perspective has zero frames (end_t={T})"

    pre = precompute_for(spec, game)
    per_t = [emit_tail(game, t, persp, spec, pre).to_dict() for t in range(T)]
    # Every key is stacked per frame over t, including scalars like value_target.
    return {key: np.stack([d[key] for d in per_t]) for key in per_t[0]}
