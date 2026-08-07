"""The golden walk: a frozen npz pair → per-frame obs, mask, and targets, pure numpy.

Mirrors `training.bc.dataset.IterableDataset._walk` frame for frame (same
init/step/build ordering, same `end_t` truncation, raw un-padded
`initial_generals`), minus the torch tensor packing — the layer the goldens
freeze beneath.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from training.bc import bfs
from training.bc.constants import W_PADDED
from training.bc.mask import build_mask
from training.bc.obs import build_obs, init_memory, step_memory
from training.bc.slots import SlotOrder
from training.bc.obs_config import ObsConfig
from training.bc.sim_types import SimFrame
from training.bc.targets.core_targets import policy_pass_target, value_target
from training.bc.targets.elim_targets import (
    make_elim_ctx,
    next_death_target,
    time_bin_targets,
)
from training.bc.visibility import compute_visibility


# The occupied config point for time_bin bin edges (ModelConfig's default).
DEFAULT_ELIM_BIN_EDGES = (10, 20, 40, 80, 160, 320, 640)


@dataclass(frozen=True)
class Frame:
    t: int
    obs: np.ndarray  # [C, H_PADDED, W_PADDED], obs_cfg dtype
    mask: np.ndarray  # [H_PADDED, W_PADDED, 8] bool
    targets: dict[str, np.ndarray | int | bool]


def walk_end_t(sim: dict[str, np.ndarray], meta: dict[str, np.ndarray], k: int) -> int:
    T = int(sim["ownership"].shape[0])
    elim_t = int(meta["elim_timestep"][k])
    return T - 1 if elim_t == -1 else min(T - 1, elim_t)


def walk_fixture(
    sim: dict[str, np.ndarray],
    meta: dict[str, np.ndarray],
    k: int,
    obs_cfg: ObsConfig,
    *,
    with_targets: bool = True,
) -> Iterator[Frame]:
    H = int(sim["map_height"])
    W = int(sim["map_width"])
    perspective_slot = int(meta["perspective_player_ids"][k])
    slot_order = SlotOrder.for_perspective(perspective_slot)
    raw_order = list(slot_order.order)

    state = init_memory(sim, perspective_slot, H, W, obs_cfg)
    cache = bfs.init_bfs_cache()

    # One ctx per elim variant: the sentinel is edges-dependent.
    tb_ctx = make_elim_ctx(sim, DEFAULT_ELIM_BIN_EDGES) if with_targets else None
    nd_ctx = make_elim_ctx(sim, None) if with_targets else None

    for t in range(walk_end_t(sim, meta, k)):
        vis = compute_visibility(sim["ownership"][t], perspective_slot, H, W)
        step_memory(state, sim, t, vis, perspective_slot, H, W)
        sim_frame = SimFrame(sim=sim, t=t, slot_order=slot_order)
        obs = build_obs(sim_frame, vis, state, cache, H, W)
        mask = build_mask(sim, t, perspective_slot, H, W)

        targets: dict[str, np.ndarray | int | bool] = {}
        if with_targets:
            assert tb_ctx is not None and nd_ctx is not None
            is_pass, flat_idx = policy_pass_target(sim, perspective_slot, t, W, W_PADDED)
            bins, alive = time_bin_targets(tb_ctx, raw_order, t)
            nxt, present, dt, removal_dt = next_death_target(nd_ctx, raw_order, t)
            targets = {
                "is_pass": is_pass,
                "action_target": flat_idx,
                "value_target": value_target(int(meta["placement"][k])),
                "alive_mask": alive,
                "elim_bin_target": bins,
                "next_elim_target": nxt,
                "present_mask": present,
                "next_elim_dt": dt,
                "next_elim_removal_dt": removal_dt,
            }
        yield Frame(t=t, obs=obs, mask=mask, targets=targets)
