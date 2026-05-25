"""World model and player perspective for the eval bot.

WorldModel owns the raw sim state and per-tick update pipeline.
PlayerView is the fog-filtered perspective that decision logic receives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import sim_core

# TODO: cross-package dep on training internals — extract shared
# fog-tracking module if this coupling becomes painful.
from bc import obs as bc_obs, visibility
from bc.obs import MemoryState

from self_play.agent import pad_initial_generals
from self_play.sim_adapter import _capture_events_to_array

P = 8  # fixed slot count the obs encoder was trained on


@dataclass(frozen=True)
class PlayerView:
    """Everything the bot can legally see this tick.

    Decision functions receive this, nothing else.
    """
    own: np.ndarray       # [H, W] int8 — fog-filtered (-2 = no info)
    arm: np.ndarray       # [H, W] int16 — fog-filtered (0 behind fog)
    vis: np.ndarray       # [H, W] bool
    passable: np.ndarray  # [H*W] bool

    my_slot: int
    gen_flat: int
    H: int
    W: int
    timestep: int

    land: np.ndarray      # [P] int32 — scoreboard
    army: np.ndarray      # [P] int64 — scoreboard
    alive: np.ndarray     # [P] bool

    # TODO: cherry-pick specific MemoryState fields into PlayerView instead
    # of passing the full object. Currently decision logic can accidentally
    # reach raw state through MemoryState's internal buffers. Acceptable for
    # v1 but should be tightened before the codebase grows.
    mem: MemoryState


class WorldModel:
    """Owns raw sim state, runs per-tick updates, produces PlayerView."""

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
        self._memory: MemoryState | None = None

    def init_for_game(self, state: sim_core.State, static: Any) -> None:
        H, W = static.map_height, static.map_width
        HW = H * W
        T = self.max_ticks_hint

        self._H, self._W = H, W

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
            "initial_cities": np.asarray(
                static.initial_cities, dtype=np.int32,
            ),
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

    def update(self, state: sim_core.State, static: Any) -> PlayerView:
        assert self._sim is not None, "call init_for_game first"
        assert self._memory is not None
        assert self._ownership_buf is not None
        assert self._armies_buf is not None

        t = state.timestep
        H, W = self._H, self._W

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

        # 3. compute visibility
        own_flat = self._ownership_buf[t]
        arm_flat = self._armies_buf[t]
        vis = visibility.compute_visibility(
            own_flat, self.perspective_slot, H, W,
        )

        # 4. backfill scoreboard into memory history
        land = np.zeros(P, dtype=np.int32)
        army = np.zeros(P, dtype=np.int64)
        for p in range(P):
            mask = own_flat == p
            land[p] = mask.sum()
            army[p] = (arm_flat * mask).sum()
        self._memory.land_count_history[t] = land
        self._memory.army_count_history[t] = army

        # 5. advance fog-tracking memory
        bc_obs.step_memory(
            self._memory, self._sim, t, vis,
            self.perspective_slot, H, W, P,
        )

        # 6. fog-filtered arrays
        own_2d = own_flat.reshape(H, W)
        arm_2d = arm_flat.reshape(H, W)
        fog_own = np.where(vis, own_2d, np.int8(-2))
        fog_arm = np.where(vis, arm_2d, np.int16(0))

        # 7. passable mask
        passable = known_passable_mask(self._memory, H, W)

        alive = np.zeros(P, dtype=bool)
        for p in range(state.num_players):
            alive[p] = state.alive[p]

        return PlayerView(
            own=fog_own,
            arm=fog_arm,
            vis=vis,
            passable=passable,
            my_slot=self.perspective_slot,
            gen_flat=self._gen_flat,
            H=H,
            W=W,
            timestep=t,
            land=land,
            army=army,
            alive=alive,
            mem=self._memory,
        )


def known_passable_mask(
    mem: MemoryState, H: int, W: int,
) -> np.ndarray:
    """Flat bool [H*W] passability mask for the bot's BFS.

    v1 policy (spec §7): structures in fog are treated as impassable
    (mountain vs. city is indistinguishable behind fog). No
    city-traversability ratio — simpler than the NN's
    compute_known_passable.
    """
    structures_in_fog = (
        mem.is_structure
        & ~mem.known_mountain
        & ~mem.known_city
        & ~mem.known_general
    )
    known_neutral_city = mem.known_city & (mem.last_seen_owner == -1)
    # v1: bot never captures neutral cities, so they're impassable like mountains.
    impassable = mem.known_mountain | structures_in_fog | known_neutral_city
    return (~impassable).reshape(H * W)
