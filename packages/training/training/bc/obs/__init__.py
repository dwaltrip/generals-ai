"""
Observation-tensor construction — the policy/value network's input at one
(game, perspective, timestep) snapshot.

The package owns three things, split across submodules:
  - `geometry` — pure spatial / slot helpers: `canonical_slot_order` (the
    Phase-1 perspective→slot-0 pin), Moore dilation, general padding, the
    perspective-filtered board view.
  - `memory` — `MemoryState` + `init_memory` / `step_memory`: per-perspective
    running memory, advanced once per tick, plus the step-time dense-history
    encoders and the cat-5 BFS passability policy.
  - `build` — `build_obs`: a pure read of (sim, state, vis, bfs_cache) →
    `[obs_channel_count(n), H_PADDED, W_PADDED]`, assembled from the `channels`
    `_cat_*` builders.

The channel count is `obs_channel_count(state.obs_cfg.dense_history_n)`: a fixed
base block plus a `2 * n` dense-history tail. The obs-encoder config rides on
`MemoryState` (set at `init_memory`), so `build_obs` reads `n` off the state it
already receives rather than taking another parameter.

This `__init__` re-exports the package's public surface; importers use
`from training.bc.obs import build_obs, MemoryState, ...` as before.
"""

from training.bc.obs.build import build_obs
from training.bc.obs.geometry import (
    PerspectiveView,
    canonical_slot_order,
    make_perspective_view,
    pad_initial_generals,
)
from training.bc.obs.memory import (
    MemoryState,
    compute_known_passable,
    init_memory,
    init_memory_common,
    init_memory_live_fog_only,
    scoreboard_row,
    step_memory,
)

__all__ = [
    "MemoryState",
    "PerspectiveView",
    "build_obs",
    "canonical_slot_order",
    "compute_known_passable",
    "init_memory",
    "init_memory_common",
    "init_memory_live_fog_only",
    "make_perspective_view",
    "pad_initial_generals",
    "scoreboard_row",
    "step_memory",
]
