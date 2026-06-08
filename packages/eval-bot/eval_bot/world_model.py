"""World model and player perspective for the eval bot.

WorldModel owns the raw sim state and per-tick update pipeline.
PlayerView is the fog-filtered perspective that decision logic receives.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

# TODO: cross-package dep on training internals — extract shared
# fog-tracking module if this coupling becomes painful.
from bc import obs as bc_obs
from bc import visibility
from training.bc.obs import MemoryState, pad_initial_generals
from eval_bot.bfs import bfs_distances
from eval_bot.bot_config import BotConfig
from game_runner.sim_adapter import capture_events_to_array
from game_types import StaticMap
import sim_core


P = 8  # fixed slot count the obs encoder was trained on


class ThreatEntry(NamedTuple):
    pos: int
    army: int
    dist_to_general: int


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

    contact_set: frozenset[int]
    threat_windows: dict[int, deque[ThreatEntry]]
    incursion_threats: dict[int, bool]
    # TODO: not read yet — ThreatScore input, needed for de-escalation (spec §6c.3)
    has_recent_attack: dict[int, bool]


class WorldModel:
    """Owns raw sim state, runs per-tick updates, produces PlayerView."""

    def __init__(
        self,
        perspective_slot: int,
        cfg: BotConfig,
    ):
        self.perspective_slot = perspective_slot
        self.cfg = cfg

        self._H: int = 0
        self._W: int = 0
        self._gen_flat: int = 0
        self._ownership: list[np.ndarray] | None = None
        self._armies: list[np.ndarray] | None = None
        self._sim: dict[str, Any] | None = None
        self._memory: MemoryState | None = None

        # per-contact-player sliding buffer of the largest visible enemy stack
        self._threat_windows: dict[int, deque[ThreatEntry]] = {}
        # player slot → bool [H, W] snapshot of (ownership == my_slot) taken when
        # that player's threat window opens; used to detect tile ownership flips for IncursionThreat
        self._eating_baselines: dict[int, np.ndarray] = {}

    def init_for_game(self, state: sim_core.State, map_data: StaticMap) -> None:
        H, W = map_data.map_height, map_data.map_width

        self._H, self._W = H, W

        # update() is the single owner of per-tick history — it appends one
        # snapshot + scoreboard row per tick, starting at tick 0. Start empty;
        # self._sim and self._memory's history alias these lists.
        self._ownership = []
        self._armies = []

        ig_padded = pad_initial_generals(
            map_data.initial_generals, self.perspective_slot,
        )
        self._gen_flat = int(ig_padded[self.perspective_slot])

        self._sim = {
            "ownership": self._ownership,
            "armies": self._armies,
            "mountains": np.asarray(map_data.mountains, dtype=np.int32),
            "initial_cities": np.asarray(
                map_data.initial_cities, dtype=np.int32,
            ),
            "initial_generals": ig_padded,
            "cities": np.asarray(state.cities, dtype=np.int32),
            "cities_present_at": np.asarray(
                state.cities_present_at, dtype=np.int32,
            ),
            "capture_events": capture_events_to_array(state.capture_events),
        }
        # Fog-only memory: the eval bot reads MemoryState's fog/scoreboard
        # fields, never the dense-history obs channels, so the obs-encoder
        # config is immaterial here (the wrapper fills the default).
        self._memory = bc_obs.init_memory_live_fog_only(
            self._sim, self.perspective_slot, H, W, P,
        )

        self._threat_windows = {}
        self._eating_baselines = {}

    def update(self, state: sim_core.State, map_data: StaticMap) -> PlayerView:
        assert self._sim is not None, "call init_for_game first"
        assert self._memory is not None
        assert self._ownership is not None
        assert self._armies is not None

        t = state.timestep
        H, W = self._H, self._W

        # update() is the single owner of per-tick history: exactly one row per
        # tick, appended in order, so `_count_history[t]` — read by absolute t in
        # attack.py — is the scoreboard for snapshot t. Assert the contract so an
        # out-of-order or repeat call fails loudly rather than silently shifting
        # the history a tick stale.
        assert t == len(self._ownership), (
            f"update() out of order: {len(self._ownership)} ticks recorded, got t={t}"
        )

        # 1. append latest snapshot
        own_flat = np.asarray(state.snapshots_ownership[t], dtype=np.int8)
        arm_flat = np.asarray(state.snapshots_armies[t], dtype=np.int16)
        self._ownership.append(own_flat)
        self._armies.append(arm_flat)

        # 2. refresh dynamic sim fields
        self._sim["cities"] = np.asarray(state.cities, dtype=np.int32)
        self._sim["cities_present_at"] = np.asarray(
            state.cities_present_at, dtype=np.int32,
        )
        self._sim["capture_events"] = capture_events_to_array(
            state.capture_events,
        )

        # 3. compute visibility
        vis = visibility.compute_visibility(
            own_flat, self.perspective_slot, H, W,
        )

        # 4. append scoreboard row
        land, army = bc_obs.scoreboard_row(own_flat, arm_flat, P)
        self._memory.land_count_history.append(land)
        self._memory.army_count_history.append(army)

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

        # 8-10. threat detection
        contact_set = self._compute_contact_set()
        self._update_threat_windows(fog_own, fog_arm, passable, contact_set)
        incursion_threats = self._evaluate_incursion_threats(fog_own)
        has_recent_attack = self._compute_has_recent_attack(fog_own)

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
            contact_set=frozenset(contact_set),
            threat_windows=self._threat_windows,
            incursion_threats=incursion_threats,
            has_recent_attack=has_recent_attack,
        )

    def _compute_contact_set(self) -> set[int]:
        assert self._memory is not None
        mem = self._memory
        contacted_players: set[int] = set()
        for p in range(P):
            if mem.opp_contacted[p] and mem.opp_captured_by[p] == -1:
                contacted_players.add(p)
        return contacted_players

    def _update_threat_windows(
        self,
        fog_own: np.ndarray,
        fog_arm: np.ndarray,
        passable: np.ndarray,
        contact_set: set[int],
    ) -> None:
        H, W = self._H, self._W

        # clear windows for players no longer in contact
        for p in list(self._threat_windows):
            if p not in contact_set:
                self._threat_windows.pop(p, None)
                self._eating_baselines.pop(p, None)

        for p in contact_set:
            visible_p = fog_own == p
            if not visible_p.any():
                self._threat_windows.pop(p, None)
                self._eating_baselines.pop(p, None)
                continue

            # largest visible stack for this player
            p_armies = np.where(visible_p, fog_arm, np.int16(0))
            best_flat = int(p_armies.argmax())
            best_army = int(p_armies.flat[best_flat])
            best_r, best_c = divmod(best_flat, W)

            # identity check — stack moves at most 1 cardinal tile per tick
            window = self._threat_windows.get(p)
            if window is not None and len(window) > 0:
                prev = window[-1]
                prev_r, prev_c = divmod(prev.pos, W)
                if abs(best_r - prev_r) + abs(best_c - prev_c) > 1:
                    self._threat_windows.pop(p, None)
                    self._eating_baselines.pop(p, None)

            # snapshot eating baseline when window opens fresh
            if p not in self._threat_windows:
                self._threat_windows[p] = deque(maxlen=self.cfg.THREAT_WINDOW_LEN)
                self._eating_baselines[p] = (
                    fog_own == self.perspective_slot
                ).copy()

            dist = bfs_distances(best_flat, passable, H, W)
            dist_to_gen = int(dist[self._gen_flat])
            if dist_to_gen < 0:
                continue

            self._threat_windows[p].append(
                ThreatEntry(pos=best_flat, army=best_army, dist_to_general=dist_to_gen),
            )

    def _evaluate_incursion_threats(
        self, fog_own: np.ndarray,
    ) -> dict[int, bool]:
        H, W = self._H, self._W
        is_threat_by_player: dict[int, bool] = {}

        for p, window in self._threat_windows.items():
            if len(window) < 2:
                is_threat_by_player[p] = False
                continue

            # closing: moving average of distance deltas
            deltas = [
                window[i].dist_to_general - window[i - 1].dist_to_general
                for i in range(1, len(window))
            ]
            closing = (sum(deltas) / len(deltas)) < self.cfg.CLOSING_THRESHOLD

            # eating: tile flips within manhattan radius of threat
            threat_pos = window[-1].pos
            threat_r, threat_c = divmod(threat_pos, W)
            baseline = self._eating_baselines.get(p)
            if baseline is None:
                is_threat_by_player[p] = False
                continue

            eaten = 0
            eat_r = self.cfg.EATING_RADIUS
            r_lo = max(0, threat_r - eat_r)
            r_hi = min(H, threat_r + eat_r + 1)
            c_lo = max(0, threat_c - eat_r)
            c_hi = min(W, threat_c + eat_r + 1)
            for r in range(r_lo, r_hi):
                for c in range(c_lo, c_hi):
                    if abs(r - threat_r) + abs(c - threat_c) > eat_r:
                        continue
                    if baseline[r, c] and fog_own[r, c] == p:
                        eaten += 1

            eating = eaten >= self.cfg.EATING_MIN
            is_threat_by_player[p] = closing and eating

        return is_threat_by_player

    # TODO: not read yet — ThreatScore input, needed for de-escalation (spec §6c.3)
    def _compute_has_recent_attack(
        self, fog_own: np.ndarray,
    ) -> dict[int, bool]:
        """Has player p captured any of our tiles since their threat window opened?"""
        result: dict[int, bool] = {}
        for p, baseline in self._eating_baselines.items():
            result[p] = bool((baseline & (fog_own == p)).any())
        return result


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
