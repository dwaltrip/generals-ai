"""ModelAgent — one perspective's brain for a self-play game.

Holds the model, per-player MemoryState / BFSCache, a pre-allocated
growing `sim` dict (`[max_turns, HW]` ownership/armies buffers we fill
row-by-row), and the canonical-slot ordering for this perspective.

The act() loop on each tick:
  1. Append the latest sim snapshot to the growing buffers
  2. Refresh dynamic sim fields (cities, capture_events)
  3. Update MemoryState's scoreboard row for the current tick
     (init_memory pre-computes the full game's scoreboard up-front; for
      online play we backfill rows as we step)
  4. Compute the perspective's visibility
  5. step_memory → build_obs → build_mask
  6. Model forward, masked argmax over policy logits, pass-head sign
  7. actions.decode → wire move (source, dest, is50)

Two ModelAgent instances drive a 2-player game; they share the same
underlying nn.Module (same weights = self-play). Each instance carries
its own perspective state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

import sim_core

from bc import actions, bfs as bc_bfs, mask as bc_mask, obs as bc_obs, visibility
from bc.constants import W_PADDED
from bc.loss import flatten_policy_logits
from bc.model import BCModel

from self_play import sim_adapter


P = 8  # fixed slot count the model + obs encoder were trained on


def load_checkpoint(path: str | Path, device: torch.device) -> BCModel:
    """Construct a BCModel and load weights from a `.pt` state-dict file."""
    model = BCModel()
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ModelAgent:
    """One perspective. Owns its memory, sim dict, and an opaque reference
    to the (shared) model. Stateless across games — re-init via
    `init_for_game(...)` each new game.
    """

    def __init__(
        self,
        model: BCModel,
        perspective_slot: int,
        device: torch.device,
        max_turns_hint: int = 1000,
    ):
        self.model = model
        self.device = device
        self.perspective_slot = perspective_slot
        self.opp_slots = bc_obs.canonical_slot_order(perspective_slot, P)[1:]
        self.max_turns_hint = max_turns_hint

        # Filled by init_for_game
        self._H: int = 0
        self._W: int = 0
        self._ownership_buf: np.ndarray | None = None
        self._armies_buf: np.ndarray | None = None
        self._sim: dict[str, Any] | None = None
        self._memory: bc_obs.MemoryState | None = None
        self._bfs_cache: bc_bfs.BFSCache | None = None

    def init_for_game(self, state: sim_core.State, static: Any) -> None:
        H, W = static.map_height, static.map_width
        HW = H * W
        T = self.max_turns_hint

        # Pre-allocate ownership/armies buffers. init_memory's scoreboard
        # precompute will run over all T rows; we leave rows 1..T-1 at the
        # default fill (-1 for ownership = neutral, 0 for armies) so the
        # bogus scoreboard counts for unreached rows are all-zero. Row 0
        # gets the real initial snapshot.
        self._H, self._W = H, W
        self._ownership_buf = np.full((T, HW), -1, dtype=np.int8)
        self._armies_buf = np.zeros((T, HW), dtype=np.int16)
        self._ownership_buf[0] = state.snapshots_ownership[0]
        self._armies_buf[0] = state.snapshots_armies[0]

        self._sim = {
            "ownership": self._ownership_buf,
            "armies": self._armies_buf,
            "mountains": np.asarray(static.mountains, dtype=np.int32),
            "initial_cities": np.asarray(static.initial_cities, dtype=np.int32),
            "initial_generals": np.asarray(static.initial_generals, dtype=np.int32),
            "cities": np.asarray(state.cities, dtype=np.int32),
            "cities_present_at": np.asarray(state.cities_present_at, dtype=np.int32),
            "capture_events": sim_adapter._capture_events_to_array(state.capture_events),
        }
        self._memory = bc_obs.init_memory(self._sim, self.perspective_slot, H, W, P)
        self._bfs_cache = bc_bfs.init_bfs_cache(P)

    def act(self, state: sim_core.State, static: Any) -> tuple[int, int, int]:
        """Run inference for the current tick and return a wire move
        `(source, dest, is50)`. `(-1, -1, -1)` means "pass" — the driver
        loop should skip these (don't submit to sim_core.step_tick).
        """
        assert self._sim is not None, "call init_for_game first"
        assert self._memory is not None
        assert self._bfs_cache is not None
        assert self._ownership_buf is not None
        assert self._armies_buf is not None

        t = state.timestep
        H, W = self._H, self._W

        # 1. Append latest snapshot row
        self._ownership_buf[t] = state.snapshots_ownership[t]
        self._armies_buf[t] = state.snapshots_armies[t]

        # 2. Refresh dynamic sim fields. Cities can grow mid-game (general→
        # city on capture); capture_events grows whenever a capture fires.
        # The numpy arrays we built at init were snapshots; rebuild.
        self._sim["cities"] = np.asarray(state.cities, dtype=np.int32)
        self._sim["cities_present_at"] = np.asarray(
            state.cities_present_at, dtype=np.int32,
        )
        self._sim["capture_events"] = sim_adapter._capture_events_to_array(
            state.capture_events,
        )

        # 3. Backfill scoreboard row for t (init_memory left it at zeros).
        own_t = self._ownership_buf[t]
        arm_t = self._armies_buf[t]
        for p in range(P):
            mask = own_t == p
            self._memory.land_count_history[t, p] = mask.sum()
            self._memory.army_count_history[t, p] = (arm_t * mask).sum()

        # 4. Visibility for this perspective
        vis = visibility.compute_visibility(own_t, self.perspective_slot, H, W)

        # 5. Advance memory + invalidate BFS cache if the known-passable
        # graph grew this tick.
        graph_grew = bc_obs.step_memory(
            self._memory, self._sim, t, vis, self.perspective_slot, H, W, P,
        )
        if graph_grew:
            self._bfs_cache.invalidate_graph()

        # 6. Build the obs tensor + legality mask
        obs_np = bc_obs.build_obs(
            self._sim, t, self.perspective_slot, self.opp_slots,
            vis, self._memory, self._bfs_cache, H, W,
        )
        mask_np = bc_mask.build_mask(self._sim, t, self.perspective_slot, H, W)

        # 7. Forward + masked argmax
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(obs_tensor)
        masked_logits = flatten_policy_logits(out["policy_logits"], mask_tensor)

        is_pass = bool((out["pass_logit"] > 0).item())
        if is_pass or not mask_tensor.any():
            # `not mask.any()` handles the degenerate "no legal moves" case
            # (e.g., very early ticks where armies are still <2). Even if
            # the pass head said act, the model has nothing legal to pick.
            return -1, -1, -1

        flat_idx = int(masked_logits.argmax(dim=1).item())
        return actions.decode(is_pass=False, flat_idx=flat_idx,
                              W_unpadded=W, W_padded=W_PADDED)
