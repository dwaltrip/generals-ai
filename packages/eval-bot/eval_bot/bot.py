"""EvalBot — hand-coded greedy agent for evaluating NN training progress.

Same duck-typed interface as ModelAgent in self_play.agent:
  - init_for_game(state, static) — called once per game
  - act(state, static) -> (src, dst, is50) — called once per tick

Milestone 1: ExpandOrExplore only. Future milestones add Defend,
Kill-shot, Attack gates per the design doc.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import sim_core

# TODO: cross-package dep on training internals (bc.obs, bc.visibility).
# Extract shared fog-tracking module if this coupling becomes painful.
from bc import obs as bc_obs, visibility

from self_play.agent import pad_initial_generals
from self_play.sim_adapter import _capture_events_to_array

from eval_bot.expand import pick_expand_or_explore
from eval_bot.world_model import known_passable_mask

P = 8  # fixed slot count the obs encoder was trained on


class EvalBot:
    def __init__(
        self,
        perspective_slot: int,
        max_ticks_hint: int = 2000,
    ):
        self.perspective_slot = perspective_slot
        self.max_ticks_hint = max_ticks_hint

        self._H: int = 0
        self._W: int = 0
        self._gen_flat: int = 0
        self._ownership_buf: np.ndarray | None = None
        self._armies_buf: np.ndarray | None = None
        self._sim: dict[str, Any] | None = None
        self._memory: bc_obs.MemoryState | None = None

        self.n_passed = 0
        self.n_moved = 0
        self.n_no_move = 0

    def init_for_game(self, state: sim_core.State, static: Any) -> None:
        H, W = static.map_height, static.map_width
        HW = H * W
        T = self.max_ticks_hint

        self._H, self._W = H, W

        # Pre-allocate ownership/armies buffers — same pattern as ModelAgent.
        # TODO: extract shared per-tick world-model update
        # (duplicated with ModelAgent / EvalBot).
        self._ownership_buf = np.full((T, HW), -1, dtype=np.int8)
        self._armies_buf = np.zeros((T, HW), dtype=np.int16)
        self._ownership_buf[0] = state.snapshots_ownership[0]
        self._armies_buf[0] = state.snapshots_armies[0]

        ig_padded = pad_initial_generals(
            static.initial_generals, self.perspective_slot,
        )
        self._gen_flat = int(ig_padded[self.perspective_slot])

        self._sim = {
            "ownership": self._ownership_buf,
            "armies": self._armies_buf,
            "mountains": np.asarray(static.mountains, dtype=np.int32),
            "initial_cities": np.asarray(static.initial_cities, dtype=np.int32),
            "initial_generals": ig_padded,
            "cities": np.asarray(state.cities, dtype=np.int32),
            "cities_present_at": np.asarray(
                state.cities_present_at, dtype=np.int32,
            ),
            "capture_events": _capture_events_to_array(state.capture_events),
        }
        self._memory = bc_obs.init_memory(
            self._sim, self.perspective_slot, H, W, P,
        )

        self.n_passed = 0
        self.n_moved = 0
        self.n_no_move = 0

    def act(
        self, state: sim_core.State, static: Any,
    ) -> tuple[int, int, int]:
        """Decide one move for the current tick.

        Returns (source, dest, is50) or (-1, -1, -1) for pass.
        """
        assert self._sim is not None, "call init_for_game first"
        assert self._memory is not None
        assert self._ownership_buf is not None
        assert self._armies_buf is not None

        t = state.timestep
        H, W = self._H, self._W

        # --- per-tick bookkeeping (mirrors ModelAgent.act) ---
        # TODO: extract shared per-tick world-model update
        # (duplicated with ModelAgent / EvalBot).

        # 1. append latest snapshot
        self._ownership_buf[t] = state.snapshots_ownership[t]
        self._armies_buf[t] = state.snapshots_armies[t]

        # 2. refresh dynamic sim fields
        self._sim["cities"] = np.asarray(state.cities, dtype=np.int32)
        self._sim["cities_present_at"] = np.asarray(
            state.cities_present_at, dtype=np.int32,
        )
        self._sim["capture_events"] = _capture_events_to_array(
            state.capture_events,
        )

        # 3. backfill scoreboard row
        own_flat = self._ownership_buf[t]
        arm_flat = self._armies_buf[t]
        for p in range(P):
            mask = own_flat == p
            self._memory.land_count_history[t, p] = mask.sum()
            self._memory.army_count_history[t, p] = (arm_flat * mask).sum()

        # 4. visibility
        vis = visibility.compute_visibility(
            own_flat, self.perspective_slot, H, W,
        )

        # 5. advance fog-tracking memory
        bc_obs.step_memory(
            self._memory, self._sim, t, vis, self.perspective_slot, H, W, P,
        )

        # --- bot decision ---
        own_2d = own_flat.reshape(H, W)
        arm_2d = arm_flat.reshape(H, W)
        passable = known_passable_mask(self._memory, H, W)

        move = pick_expand_or_explore(
            own_2d, arm_2d, self.perspective_slot, self._memory,
            passable, self._gen_flat, H, W,
        )

        if move is None:
            self.n_no_move += 1
            return (-1, -1, -1)

        self.n_moved += 1
        return move
