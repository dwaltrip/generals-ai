from __future__ import annotations

import numpy as np

from training.bc.aux_heads.elim_head_meta import ElimHeadVariant
from training.bc.constants import W_PADDED
from training.bc.mask import build_mask
from training.bc.player_status import make_alive_mask, precompute_player_status
from training.bc.sim_types import PerspectiveMeta
from training.bc.targets.core_targets import policy_pass_target, value_target
from training.bc.targets.elim_targets import (
    make_elim_ctx,
    next_death_target,
    time_bin_targets,
)
from training.goldens.registry import TargetsEntry

# TODO: figure out dtype declarations for the goldens
# TODO(sketch): value_target scalar vs per-frame (8.12-2 build-time pick)


def compute_targets(
    sim: dict[str, np.ndarray],
    persp: PerspectiveMeta,
    entry: TargetsEntry,
) -> dict[str, np.ndarray]:
    H = int(sim["map_height"])
    W = int(sim["map_width"])
    slot = persp.slot
    raw_order = list(persp.slot_order.order)
    variant = entry.cfg.elim_variant
    T = persp.end_t

    elim = make_elim_ctx(sim, entry.cfg.elim_bin_edges) if variant is not None else None
    status = None
    if entry.emit_alive_mask:
        status = elim.player_status if elim is not None else precompute_player_status(sim)

    out: dict[str, np.ndarray] = {}
    out["mask"] = np.stack([build_mask(sim, t, slot, H, W) for t in range(T)])

    is_pass = np.empty(T, dtype=np.bool_)
    action = np.empty(T, dtype=np.int64)
    for t in range(T):
        p, idx = policy_pass_target(sim, slot, t, W, W_PADDED)
        is_pass[t] = p
        action[t] = idx
    out["is_pass"] = is_pass
    out["action_target"] = action
    out["value_target"] = np.asarray(value_target(persp.placement), dtype=np.int64)

    if status is not None:
        out["alive_mask"] = np.stack(
            [make_alive_mask(status, raw_order, t) for t in range(T)]
        )

    if variant == ElimHeadVariant.TIME_BIN:
        assert elim is not None
        out["elim_bin_target"] = np.stack(
            [time_bin_targets(elim, raw_order, t)[0] for t in range(T)]
        )
    elif variant == ElimHeadVariant.NEXT_DEATH:
        assert elim is not None
        nxt = np.empty(T, dtype=np.int64)
        dt = np.empty(T, dtype=np.int64)
        present = np.empty((T, 8), dtype=np.bool_)
        removal_dt = np.empty((T, 8), dtype=np.int64)
        for t in range(T):
            n, p_mask, d, r = next_death_target(elim, raw_order, t)
            nxt[t] = n
            present[t] = p_mask
            dt[t] = d
            removal_dt[t] = r
        out["next_elim_target"] = nxt
        out["present_mask"] = present
        out["next_elim_dt"] = dt
        out["next_elim_removal_dt"] = removal_dt

    return {key: out[key] for key in entry.keys}
