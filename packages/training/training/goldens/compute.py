from __future__ import annotations

import numpy as np

from training.bc.datapipe.emit import emit_tail
from training.bc.datapipe.emit_spec import PartialEmitSpec
from training.bc.datapipe.precompute import precompute_for
from training.bc.datapipe.sim_types import PerspectiveMeta, SimGame
from training.bc.datapipe.walk import walk
from training.bc.obs_config import ObsConfig
from training.goldens.hashes import ObsDigest, ObsHasher


# The obs goldens guard the bytes of the obs tensor as produced during training.
# Hashes per frame and per channel are enough to robustly detect changes and tell
# us where to look (which ticks and channels were affected).
# Higher fidelity alternatives were considered (storing the full tensor or hashing
# frames x channels), but the storage cost would be burdensome for a small gain
# (e.g. knowing the exact frame on which a channel changed).
def compute_obs(game: SimGame, persp: PerspectiveMeta, cfg: ObsConfig) -> ObsDigest:
    hasher = ObsHasher()
    for frame in walk(game, persp, cfg):
        hasher.add(frame.obs)
    return hasher.digest()


def compute_supervision(
    game: SimGame,
    persp: PerspectiveMeta,
    spec: PartialEmitSpec,
) -> dict[str, np.ndarray]:
    assert persp.end_t > 0, f"perspective has zero frames (end_t={persp.end_t})"

    pre = precompute_for(spec, game)
    per_t = [emit_tail(game, t, persp, spec, pre).to_dict() for t in range(persp.end_t)]
    # Every key is stacked per frame over t, including scalars like value_target.
    return {key: np.stack([d[key] for d in per_t]) for key in per_t[0]}
