from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from training.bc import bfs
from training.bc.mask import build_mask
from training.bc.obs import build_obs, init_memory, step_memory
from training.bc.obs_config import ObsConfig
from training.bc.datapipe.sim_types import PerspectiveMeta, SimFrame
from training.bc.visibility import compute_visibility


@dataclass(frozen=True)
class WalkFrame:
    t: int
    obs: np.ndarray
    mask: np.ndarray


def walk(
    sim: dict[str, np.ndarray],
    perspective: PerspectiveMeta,
    obs_cfg: ObsConfig,
) -> Iterator[WalkFrame]:
    H = int(sim["map_height"])
    W = int(sim["map_width"])
    slot = perspective.slot

    state = init_memory(sim, slot, H, W, obs_cfg)
    cache = bfs.init_bfs_cache()

    for t in range(perspective.end_t):
        vis = compute_visibility(sim["ownership"][t], slot, H, W)
        step_memory(state, sim, t, vis, slot, H, W)
        sim_frame = SimFrame(sim=sim, t=t, slot_order=perspective.slot_order)
        obs = build_obs(sim_frame, vis, state, cache, H, W)
        mask = build_mask(sim, t, slot, H, W)
        yield WalkFrame(t=t, obs=obs, mask=mask)
