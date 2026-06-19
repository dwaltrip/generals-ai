"""`build_obs` — the obs-tensor assembler.

Pure read of (sim, state, vis, bfs_cache) → `[obs_channel_count(n), H_PADDED,
W_PADDED]`. The channel set is grouped by category in `channels` (one `_cat_*`
per group); this module orchestrates them and stacks the result. The stacking
order below is the source of truth for layout (the names in `bc.constants` are
reference docs).
"""

from __future__ import annotations

import numpy as np

from training.bc import bfs
from training.bc.constants import H_PADDED, W_PADDED, obs_channel_count
from training.bc.obs.channels import (
    _cat_bfs,
    _cat_contact_capture,
    _cat_dense_history,
    _cat_memory,
    _cat_opp_broadcast,
    _cat_persistent_map,
    _cat_scoreboard,
    _cat_self_broadcast,
    _cat_visibility,
    _cat_visible_state,
)
from training.bc.obs.memory import MemoryState
from training.shared.timing import timer


# `ObsConfig.obs_dtype` ("fp32"/"fp16") → the numpy dtype of the assembled obs
# buffer. fp16 halves every host-side byte on the worker→GPU handoff path.
_NP_OBS_DTYPE = {"fp32": np.float32, "fp16": np.float16}


def build_obs(
    sim: dict[str, np.ndarray],
    t: int,
    perspective_slot: int,
    opp_slots: list[int],
    vis: np.ndarray,
    state: MemoryState,
    bfs_cache: bfs.BFSCache,
    H: int,
    W: int,
) -> np.ndarray:
    """
    Build the obs tensor for one (game, perspective, tick). Channel count is
    `obs_channel_count(state.obs_cfg.dense_history_n)` (96 at the default n=5).

    Channel layout: the stack order below is the source of truth (the names in
    `bc.constants` are reference docs). Each `_cat_*` helper returns its slice
    in stack-order position; this function concatenates them, and the assert
    guards against count drift.

    `step_memory` must have already been called for this `(state, t, vis)` —
    `build_obs` is pure-read w.r.t. `state` (the only side effect is lazy
    cache fills inside `bfs.compute_or_get`, which are idempotent).
    """
    with timer.section("prelude"):
        own = sim["ownership"][t].reshape(H, W)
        armies = sim["armies"][t].reshape(H, W)

        # Cross-cat helper — bool form feeds both cat 3 (as a channel) and cat 5
        # (the BFS passability policy treats fog-structures as impassable).
        # cities_now_2d folds in any city that exists at tick t — initial cities
        # plus post-capture cities (general → city per §9 line 168), whose
        # existence is visible through fog per §5 line 87 even when the
        # perspective doesn't have direct vision of the cell.
        cities_now_flat = np.zeros(H * W, dtype=bool)
        cities_now_flat[sim["cities"][sim["cities_present_at"] <= t]] = True
        cities_now_2d = cities_now_flat.reshape(H, W)
        structures_in_fog_mask = (
            (state.is_structure | cities_now_2d)
            & ~state.known_mountain
            & ~state.known_city
            & ~state.known_general
        )

    with timer.section("channel_build"):
        channels = [
            *_cat_visibility(vis),
            *_cat_visible_state(vis, own, armies, perspective_slot, opp_slots),
            *_cat_persistent_map(state, structures_in_fog_mask),
            *_cat_memory(state, perspective_slot, opp_slots),
            *_cat_bfs(state, t, perspective_slot, opp_slots, vis, own, armies,
                      structures_in_fog_mask, bfs_cache, H, W),
            *_cat_self_broadcast(state, t, perspective_slot, H, W),
            *_cat_opp_broadcast(state, t, opp_slots, H, W),
            *_cat_scoreboard(state, t, opp_slots, H, W),
            *_cat_contact_capture(state, perspective_slot, opp_slots, H, W),
            *_cat_dense_history(state, H, W),
        ]

    n_channels = obs_channel_count(state.obs_cfg.dense_history_n)
    assert len(channels) == n_channels, (
        f"channel count mismatch: built {len(channels)}, "
        f"expected {n_channels} for dense_history_n={state.obs_cfg.dense_history_n}"
    )

    # Write each channel straight into its slot of one preallocated padded
    # buffer. Equivalent to `np.stack(channels).astype(obs_dtype)` followed by a
    # pad copy, but in a single write pass instead of three (stack gather +
    # astype copy + pad copy) — the assembly's per-sample memory traffic is the
    # cost that scales with channel count, so collapsing it matters most for
    # large dense-history depths. The pad region stays zero; the slice assignment
    # dtype-converts each (float32) channel into the buffer exactly as `.astype`
    # would — including the fp32→fp16 downcast when obs_dtype="fp16".
    with timer.section("assemble"):
        obs = np.zeros(
            (n_channels, H_PADDED, W_PADDED),
            dtype=_NP_OBS_DTYPE[state.obs_cfg.obs_dtype],
        )
        for i, ch in enumerate(channels):
            obs[i, :H, :W] = ch
    return obs
