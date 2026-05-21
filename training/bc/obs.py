"""
Observation-tensor construction — the 96-channel input to the policy/value
network at one (game, perspective, timestep) snapshot.

This module owns three things:
  - `canonical_slot_order` — pin from Phase 1 (perspective→slot-0 layout).
  - `MemoryState` + `init_memory` + `step_memory` — per-perspective running
    memory, advanced once per tick.
  - `build_obs` — pure read of (sim, state, vis, bfs_cache) → [96, H_PADDED, W_PADDED].

The channel set is grouped by category internally (named vars per channel),
then assembled in `CHANNEL_ORDER` at the bottom of `build_obs`. The named-var
shape is deliberately verbose so that the encoding for each channel is
visually self-contained and easy to audit.

--- Per-opp slot ordering contract ---

Every per-opp channel group (`opp_N_owned`, `opp_N_army_count`,
`last_seen_owner_opp_N`, BFS-to-opp-N, `opp_N_contacted`, `opp_N_captured_by`,
`opp_N_has_seen`, ...) follows the same canonical mapping: channel index `i`
for `i ∈ 1..7` corresponds to the raw slot `opp_slots[i-1]`, where
`opp_slots = canonical_slot_order(perspective)[1:]`.

Channel 0 (in any per-opp grouping that *includes* self) is always the
perspective player.

--- Knobs deferred for later tuning ---

A few obs-construction choices are knobs the model can't tell us about
until it trains. Worth coming back to:

  - BFS-policy knobs (cat 5): see the cat-5 comment for the spike's
    enemy-general / city-passability decisions, and `bfs.py`'s knob
    header for the weighted-edge upgrade path.
  - Broadcast-scalar normalization divisors (currently: hand-picked typical
    late-game values; could be game-normalized or rolling-window scaled).
  - Production-subtracted vs. raw army_delta in dense history (currently raw).
  - opp_N_city_inference (currently zero-stubbed — full encoding is the
    "infer cities from peacetime growth" heuristic from 5.05-1 §F).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from bc import bfs
from bc.constants import (
    CHANNEL_ORDER,
    CITY_TRAVERSABILITY_FACTOR,
    DENSE_HISTORY_N,
    H_PADDED,
    OBS_CHANNELS,
    W_PADDED,
)


# ---------------------------------------------------------------------------
# Canonicalization (unchanged from Phase 1)
# ---------------------------------------------------------------------------


def _moore_dilate(mask: np.ndarray) -> np.ndarray:
    """Moore-neighborhood (3x3) binary dilation of a 2D bool mask.

    Same 8-direction OR pattern as `visibility.compute_visibility`, factored
    out for reuse by `step_memory` when expanding the agent's vision of an
    opponent's tile into "opponent had vision around here."
    """
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    return out


def canonical_slot_order(perspective_slot: int, P: int = 8) -> list[int]:
    """
    Raw slot ids in canonical channel order. Channel 0 is always the perspective player.

    See the module docstring of `obs.py` (Phase 1) for the full rationale —
    in brief, this is the standard slot-canonicalization trick that turns K
    per-game perspectives into K augmented views of the same dynamics rather
    than K decorrelated input distributions. Ascending-skip ordering for
    opponents matches the non-cyclic-game literature (Dota/AlphaStar/Pluribus).

    Example for P=8:
      perspective_slot=0 → [0, 1, 2, 3, 4, 5, 6, 7]
      perspective_slot=5 → [5, 0, 1, 2, 3, 4, 6, 7]
    """
    return [perspective_slot] + [s for s in range(P) if s != perspective_slot]


# ---------------------------------------------------------------------------
# Per-perspective running memory
# ---------------------------------------------------------------------------


@dataclass
class MemoryState:
    """
    Running per-cell memory for one (game, perspective) walk. Initialized at
    t=0 by `init_memory`, advanced each tick by `step_memory`, read (without
    mutation) by `build_obs`. Lifetime is scoped to the inner k-loop of the
    dataset's per-game walk.

    Field categories:
      - Static per game: `is_structure`, `land_count_history`, `army_count_history`.
        Computed once in `init_memory`; never mutated thereafter.
      - Per-cell mutable masks: `historically_seen`, `known_mountain`,
        `known_city`, `known_general`, plus the `last_seen_*` channels and
        `turns_since_seen`.
      - Per-opponent: `opp_contacted` (binary monotonic), `opp_captured_by`
        (-1=alive, 0..7=raw slot of captor).
      - Source pins: `general_locations` (-1 = unknown).
      - Dense-history buffers: bounded deques of size DENSE_HISTORY_N + 1.
    """

    # ---- Static within a game ----
    # bool [H, W]: cells the agent knows are *some* structure (mountain, city,
    # or general) — visible terrain at t=0. Per game-mechanics, structure
    # presence is visible through fog; only the *type* is hidden.
    is_structure: np.ndarray
    # int32 [P]: flat (unpadded) cell index of each player's general; -1 if
    # the perspective hasn't observed it. Updated by step_memory on first
    # sighting. Self general is set in init_memory (always known at t=0).
    general_locations: np.ndarray
    # int32 [T, P]: per-tick land count (cells owned). Computed once in
    # init_memory. Scoreboard is global; available regardless of fog.
    land_count_history: np.ndarray
    # int64 [T, P]: per-tick total army (summed over owned cells). Same shape
    # as land_count_history; same precomputation logic.
    army_count_history: np.ndarray

    # ---- Mutable per-cell (advanced by step_memory) ----
    # bool [H, W]: cells the agent has ever had vision of. Monotonic.
    historically_seen: np.ndarray
    # bool [H, W]: structure cells confirmed to be mountains. Monotonic.
    known_mountain: np.ndarray
    # bool [H, W]: structure cells confirmed to be cities. Monotonic.
    known_city: np.ndarray
    # bool [H, W]: structure cells confirmed to be (currently-alive) generals.
    # Cleared when the general becomes a city via capture (handled in
    # step_memory).
    known_general: np.ndarray
    # int8 [H, W]: last observed owner per cell. Sentinels:
    #   -2 = never seen (default at init)
    #   -1 = neutral
    #   0..7 = raw player slot
    last_seen_owner: np.ndarray
    # int16 [H, W]: raw army count at last observation. -1 = never seen.
    last_seen_armies: np.ndarray
    # int32 [H, W]: ticks since last vision of this cell. -1 = never seen.
    turns_since_seen: np.ndarray

    # ---- Per-opponent spatial memory ----
    # bool [P, H, W]: opp_has_seen[p] = cells the perspective has ever observed
    # to be within opponent p's Moore-neighborhood vision. Whenever the
    # perspective sees a tile owned by p, the tile + its 8 neighbors are
    # marked — i.e., "p had vision around here at some point." Monotonic.
    # See 5.05-1 §3.4.2 + §I. The self-slot row is unused (kept all-False).
    opp_has_seen: np.ndarray

    # ---- Per-opponent broadcast state ----
    # bool [P]: opp_contacted[p] = True iff perspective has ever seen a cell
    # owned by p. Monotonic.
    opp_contacted: np.ndarray
    # int8 [P]: opp_captured_by[p]:
    #   -1 = p is alive
    #   0..7 = raw slot of the player that captured p
    # Per design, capture events are global (everyone sees them) regardless
    # of fog.
    opp_captured_by: np.ndarray

    # ---- Dense-history buffers (bounded deques of length DENSE_HISTORY_N+1) ----
    # Each entry is the [H, W] sim snapshot at one tick. After `step_memory`
    # at tick t, the right end is own[t] / armies[t].
    own_buf: deque = field(default_factory=lambda: deque(maxlen=DENSE_HISTORY_N + 1))
    armies_buf: deque = field(default_factory=lambda: deque(maxlen=DENSE_HISTORY_N + 1))


def init_memory(
    sim: dict[str, np.ndarray],
    perspective_slot: int,
    H: int,
    W: int,
    P: int = 8,
) -> MemoryState:
    """Build a fresh `MemoryState` for one (game, perspective) walk."""
    HW = H * W

    # The static "is this cell a structure" mask: union of mountains, initial
    # cities, and initial generals. Per game-mechanics, the perspective can
    # see structure positions from t=0 (just not their type), so this set is
    # fully known from the start of the game.
    is_structure_flat = np.zeros(HW, dtype=bool)
    is_structure_flat[sim["mountains"]] = True
    is_structure_flat[sim["initial_cities"]] = True
    is_structure_flat[sim["initial_generals"]] = True
    is_structure = is_structure_flat.reshape(H, W)

    # Self general known at t=0; opponents unknown.
    general_locations = np.full(P, -1, dtype=np.int32)
    general_locations[perspective_slot] = int(sim["initial_generals"][perspective_slot])

    # Precompute per-tick scoreboard (global info — available through fog).
    # T*P*HW work once per game; cheap relative to the per-tick obs cost.
    ownership = sim["ownership"]  # [T, HW] int8
    armies = sim["armies"]  # [T, HW] int16
    T = ownership.shape[0]
    land_count_history = np.zeros((T, P), dtype=np.int32)
    army_count_history = np.zeros((T, P), dtype=np.int64)
    for p in range(P):
        owned_mask = (ownership == p)  # [T, HW]
        land_count_history[:, p] = owned_mask.sum(axis=1)
        army_count_history[:, p] = (armies * owned_mask).sum(axis=1)

    state = MemoryState(
        is_structure=is_structure,
        general_locations=general_locations,
        land_count_history=land_count_history,
        army_count_history=army_count_history,
        historically_seen=np.zeros((H, W), dtype=bool),
        known_mountain=np.zeros((H, W), dtype=bool),
        known_city=np.zeros((H, W), dtype=bool),
        known_general=np.zeros((H, W), dtype=bool),
        last_seen_owner=np.full((H, W), -2, dtype=np.int8),
        last_seen_armies=np.full((H, W), -1, dtype=np.int16),
        turns_since_seen=np.full((H, W), -1, dtype=np.int32),
        opp_has_seen=np.zeros((P, H, W), dtype=bool),
        opp_contacted=np.zeros(P, dtype=bool),
        opp_captured_by=np.full(P, -1, dtype=np.int8),
    )

    # Mark own general as known from t=0.
    own_loc = int(general_locations[perspective_slot])
    own_r, own_c = divmod(own_loc, W)
    state.known_general[own_r, own_c] = True

    return state


def step_memory(
    state: MemoryState,
    sim: dict[str, np.ndarray],
    t: int,
    vis: np.ndarray,
    perspective_slot: int,
    H: int,
    W: int,
    P: int = 8,
) -> bool:
    """
    Advance running memory by one tick. Mutates `state` in place.

    Returns True iff the known-passable graph grew this tick (caller should
    invalidate the BFS cache via `bfs_cache.invalidate_graph()`).
    """
    own_t = sim["ownership"][t].reshape(H, W)
    armies_t = sim["armies"][t].reshape(H, W)

    graph_grew = False

    # historically_seen: monotonic OR with current vision.
    state.historically_seen |= vis

    # known_mountain: visible cells with the -2 (mountain) ownership sentinel.
    # Already assumed impassable in BFS (structures_in_fog default), so this
    # doesn't grow the passable set.
    new_mountain = vis & (own_t == -2) & ~state.known_mountain
    state.known_mountain |= new_mountain

    # Compute which cells are cities at time t. Cities never disappear once
    # they exist; `cities_present_at[i]` is the snapshot when cities[i] came
    # into being. General-to-city transitions on capture also land here.
    cities_at_t_flat = sim["cities"][sim["cities_present_at"] <= t]
    cities_mask_flat = np.zeros(H * W, dtype=bool)
    cities_mask_flat[cities_at_t_flat] = True
    cities_mask_2d = cities_mask_flat.reshape(H, W)

    # known_city: visible structure cells confirmed as cities.
    new_city = vis & cities_mask_2d & ~state.known_city
    if new_city.any():
        state.known_city |= new_city
        graph_grew = True

    # known_general: visible cells that are still active generals. A cell
    # ceases to be a general when it transitions to a city (capture event).
    general_cells_flat = np.zeros(H * W, dtype=bool)
    general_cells_flat[sim["initial_generals"]] = True
    is_general_now_flat = general_cells_flat & ~cities_mask_flat
    is_general_now_2d = is_general_now_flat.reshape(H, W)

    new_general = vis & is_general_now_2d & ~state.known_general
    if new_general.any():
        state.known_general |= new_general
        graph_grew = True
        # Pin general source locations for any newly-seen opponent generals.
        # Iterates over the (at most 8) initial_generals positions and checks
        # which ones just became visible.
        for p in range(P):
            cell = int(sim["initial_generals"][p])
            if state.general_locations[p] >= 0:
                continue  # already known
            r, c = divmod(cell, W)
            if new_general[r, c]:
                state.general_locations[p] = cell

    # Also need to clear known_general for cells that JUST transitioned from
    # general → city (the captured player's home cell). These were marked
    # `known_general=True` before; once they appear in cities-at-t, they
    # should flip to `known_city`. Handled implicitly by `new_city` above
    # if the perspective has vision; if not visible at the time of capture,
    # we leave the stale `known_general=True` until next sighting.
    # TODO: clean up when the general→city transition happens regardless of
    # vision. Under the v1 BFS policy (compute_known_passable) an enemy
    # general is impassable while an enemy city might be passable via the
    # army-ratio formula, so a stale known_general flag for a cell that's
    # actually a captured-general → city slightly over-restricts the BFS
    # graph until the perspective re-sights it.

    # Memory snapshot updates — write last-seen values only for visible cells.
    state.last_seen_owner[vis] = own_t[vis]
    state.last_seen_armies[vis] = armies_t[vis]

    # turns_since_seen: reset visible cells to 0; increment fogged-but-ever-seen.
    # Never-seen cells stay at -1.
    state.turns_since_seen[vis] = 0
    fogged_known = state.historically_seen & ~vis
    state.turns_since_seen[fogged_known] += 1

    # Contact + opp_has_seen: both derived from "agent's vision of cells
    # owned by p." Contact is the OR-reduction (have I seen any of p's
    # tiles?); opp_has_seen is the Moore-dilation (which cells were within
    # p's vision when I saw them?). One pass per opponent covers both.
    for p in range(P):
        if p == perspective_slot:
            continue
        visible_owned_by_p = vis & (own_t == p)
        if not visible_owned_by_p.any():
            continue
        state.opp_contacted[p] = True
        state.opp_has_seen[p] |= _moore_dilate(visible_owned_by_p)

    # Capture events: filter to events occurring at this tick, update
    # opp_captured_by. Global event — perspective sees all captures regardless
    # of fog (per game-mechanics: capture announcements are board-wide).
    events = sim["capture_events"]
    if events.size > 0:
        events_now = events[events[:, 0] == t]
        for ev in events_now:
            captor = int(ev[1])
            captured = int(ev[2])
            state.opp_captured_by[captured] = captor

    # Dense-history buffers: append the current tick. Bounded deque drops the
    # oldest automatically once length exceeds DENSE_HISTORY_N + 1.
    state.own_buf.append(own_t.copy())
    state.armies_buf.append(armies_t.copy())

    return graph_grew


# ---------------------------------------------------------------------------
# BFS passability policy — cat 5 helper
# ---------------------------------------------------------------------------


def compute_known_passable(
    state: MemoryState,
    t: int,
    perspective_slot: int,
    vis: np.ndarray,
    own: np.ndarray,
    armies: np.ndarray,
    structures_in_fog_mask: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """
    BFS-passability mask for the perspective player at tick `t`.

    v1 spike policy (5.20-1 §3 decisions):
      - Mountains + structures_in_fog: impassable.
      - Enemy generals: impassable. Own general: passable.
      - Own cities: always passable.
      - Non-own cities (neutral / enemy): passable iff
          `total_army > city_army * CITY_TRAVERSABILITY_FACTOR`.
        `city_army` uses `armies[t]` when visible, else `last_seen_armies`
        (always >= 0 for `known_city` cells, since the mask is set only on
        direct vision).

    Returns a flat bool `[H*W]` array — True means passable.
    """
    total_army = int(state.army_count_history[t, perspective_slot])

    own_loc = int(state.general_locations[perspective_slot])
    own_general_mask = np.zeros((H, W), dtype=bool)
    own_r, own_c = divmod(own_loc, W)
    own_general_mask[own_r, own_c] = True
    enemy_general_mask = state.known_general & ~own_general_mask

    effective_owner = np.where(vis, own, state.last_seen_owner)
    effective_army = np.where(vis, armies, state.last_seen_armies).astype(np.int32)
    passable_city_mask = state.known_city & (
        (effective_owner == perspective_slot)
        | (total_army > effective_army * CITY_TRAVERSABILITY_FACTOR)
    )
    impassable_city_mask = state.known_city & ~passable_city_mask

    assumed_impassable = (
        state.known_mountain
        | structures_in_fog_mask
        | enemy_general_mask
        | impassable_city_mask
    )
    return (~assumed_impassable).flatten()


# ---------------------------------------------------------------------------
# Channel category helpers
# ---------------------------------------------------------------------------
#
# Each `_cat_*` builds one category of the obs tensor and returns a list of
# `[H, W] float32` channels in CHANNEL_ORDER position. `build_obs` is the
# orchestrator that calls them and stacks the result. Comments retain the
# numeric category labels (Cat 1, Cat 2, ...) for cross-reference with the
# design docs (5.05-1 §F/§G, etc.) and CHANNEL_ORDER groupings.
#
# `structures_in_fog_mask` is the one cross-cat helper — it's the bool form
# of cat 3's `structures_in_fog` channel and also feeds the BFS passability
# policy in cat 5. Computed once in `build_obs` and threaded into both.


# Hand-picked broadcast-scalar normalization divisors. These are rough
# "typical late-game values" so the scalars land in roughly [0, 1] range.
# Knobs — revisit after observing real training-time distributions.
_ARMY_DIVISOR = 1000.0
_LAND_DIVISOR = 100.0
_LAND_DELTA_DIVISOR = 50.0
_LAND_DELTA_WINDOW = 10  # ticks over which opp_N_land_delta is measured


def _cat_visibility(vis: np.ndarray) -> list[np.ndarray]:
    # Cat 1: Visibility (1 channel)
    return [vis.astype(np.float32)]


def _cat_visible_state(
    vis: np.ndarray,
    own: np.ndarray,
    armies: np.ndarray,
    perspective_slot: int,
    opp_slots: list[int],
) -> list[np.ndarray]:
    # Cat 2: Visible state (9 channels). Ownership masks (self + 7 canonical
    # opponents), plus log-armies. Per the contract, channel for opp i ∈ 1..7
    # is `opp_slots[i-1]`.
    self_owned = (vis & (own == perspective_slot)).astype(np.float32)
    opp_owned = [(vis & (own == opp)).astype(np.float32) for opp in opp_slots]
    army_magnitude = np.where(
        vis, np.log1p(armies.astype(np.float32)), 0.0
    ).astype(np.float32)
    return [self_owned, *opp_owned, army_magnitude]


def _cat_persistent_map(
    state: MemoryState, structures_in_fog_mask: np.ndarray,
) -> list[np.ndarray]:
    # Cat 3: Persistent map knowledge (4 channels). Once-known-stays-known.
    # `structures_in_fog` = "I know this is a structure but I don't know its
    # type yet" — assumed-impassable for BFS purposes (cat 5).
    return [
        state.known_mountain.astype(np.float32),
        state.known_city.astype(np.float32),
        state.known_general.astype(np.float32),
        structures_in_fog_mask.astype(np.float32),
    ]


def _cat_memory(
    state: MemoryState, perspective_slot: int, opp_slots: list[int],
) -> list[np.ndarray]:
    # Cat 4: Memory of formerly-visible cells (19 channels). 9-way one-hot
    # last_seen_owner (self + 7 opp + neutral), log-scaled last_seen_armies,
    # turns_since_seen, historically_seen mask, plus 7 per-opp
    # `opp_N_has_seen` masks (5.05-1 §3.4.2 / §I).
    # `-1.0` post-log sentinel for never-seen cells; consistent pattern
    # across all log-scaled memory channels.
    last_seen_owner_self = (state.last_seen_owner == perspective_slot).astype(np.float32)
    last_seen_owner_opp = [
        (state.last_seen_owner == opp).astype(np.float32) for opp in opp_slots
    ]
    last_seen_owner_neutral = (state.last_seen_owner == -1).astype(np.float32)
    # `np.maximum(x, 0)` before log1p suppresses the "log of negative" warning
    # for cells where the -1 sentinel will be substituted by np.where anyway.
    last_seen_armies_safe = np.maximum(state.last_seen_armies, 0).astype(np.float32)
    last_seen_armies_ch = np.where(
        state.last_seen_armies >= 0, np.log1p(last_seen_armies_safe), -1.0
    ).astype(np.float32)
    turns_since_seen_safe = np.maximum(state.turns_since_seen, 0).astype(np.float32)
    turns_since_seen_ch = np.where(
        state.turns_since_seen >= 0, np.log1p(turns_since_seen_safe), -1.0
    ).astype(np.float32)
    historically_seen_ch = state.historically_seen.astype(np.float32)
    opp_has_seen = [state.opp_has_seen[opp].astype(np.float32) for opp in opp_slots]
    return [
        last_seen_owner_self, *last_seen_owner_opp, last_seen_owner_neutral,
        last_seen_armies_ch, turns_since_seen_ch, historically_seen_ch,
        *opp_has_seen,
    ]


def _cat_bfs(
    state: MemoryState,
    t: int,
    perspective_slot: int,
    opp_slots: list[int],
    vis: np.ndarray,
    own: np.ndarray,
    armies: np.ndarray,
    structures_in_fog_mask: np.ndarray,
    bfs_cache: bfs.BFSCache,
    H: int,
    W: int,
) -> list[np.ndarray]:
    # Cat 5: BFS distance-from-known-generals (8 channels). Passability is
    # computed by `compute_known_passable` — see its docstring for the v1
    # spike policy. Per-frame recompute: city passability depends on
    # tick-level values, so cross-frame BFS caching is invalidated every
    # frame in `dataset.py::__iter__`.
    known_passable_flat = compute_known_passable(
        state, t, perspective_slot, vis, own, armies,
        structures_in_fog_mask, H, W,
    )
    bfs_self = bfs.compute_or_get(
        bfs_cache, 0, int(state.general_locations[perspective_slot]),
        known_passable_flat, H, W,
    )
    bfs_opp = [
        bfs.compute_or_get(
            bfs_cache, i + 1, int(state.general_locations[opp]),
            known_passable_flat, H, W,
        )
        for i, opp in enumerate(opp_slots)
    ]
    return [bfs_self, *bfs_opp]


def _cat_self_broadcast(
    state: MemoryState,
    t: int,
    perspective_slot: int,
    max_turns: int,
    H: int,
    W: int,
) -> list[np.ndarray]:
    # Cat 6: Self broadcast scalars (3 channels). Same value at every cell
    # of the unpadded board; padding stays zero.
    self_army_count = np.full(
        (H, W),
        state.army_count_history[t, perspective_slot] / _ARMY_DIVISOR,
        dtype=np.float32,
    )
    self_land_count = np.full(
        (H, W),
        state.land_count_history[t, perspective_slot] / _LAND_DIVISOR,
        dtype=np.float32,
    )
    game_progress = np.full((H, W), t / max(1, max_turns), dtype=np.float32)
    return [self_army_count, self_land_count, game_progress]


def _cat_opp_broadcast(
    state: MemoryState, t: int, opp_slots: list[int], H: int, W: int,
) -> list[np.ndarray]:
    # Cat 7: Per-opponent broadcast scalars (14 channels).
    opp_army_count = [
        np.full((H, W), state.army_count_history[t, opp] / _ARMY_DIVISOR, dtype=np.float32)
        for opp in opp_slots
    ]
    opp_land_count = [
        np.full((H, W), state.land_count_history[t, opp] / _LAND_DIVISOR, dtype=np.float32)
        for opp in opp_slots
    ]
    return [*opp_army_count, *opp_land_count]


def _cat_scoreboard(
    state: MemoryState, t: int, opp_slots: list[int], H: int, W: int,
) -> list[np.ndarray]:
    # Cat 8: Scoreboard-derived broadcasts (14 channels).
    # opp_N_city_inference is the "infer cities from peacetime growth"
    # heuristic — TODO, currently emits zero. opp_N_land_delta is a simple
    # K-tick land-count delta.
    opp_city_inference = [np.zeros((H, W), dtype=np.float32) for _ in opp_slots]  # TODO: 5.05-1 §F encoding
    t_prev = max(0, t - _LAND_DELTA_WINDOW)
    opp_land_delta = [
        np.full(
            (H, W),
            (state.land_count_history[t, opp] - state.land_count_history[t_prev, opp])
                / _LAND_DELTA_DIVISOR,
            dtype=np.float32,
        )
        for opp in opp_slots
    ]
    return [*opp_city_inference, *opp_land_delta]


def _cat_contact_capture(
    state: MemoryState, perspective_slot: int, opp_slots: list[int], H: int, W: int,
) -> list[np.ndarray]:
    # Cat 9: Contact & capture (14 channels). opp_N_captured_by is integer-
    # encoded, with values remapped from raw slot to canonical channel index
    # (so the model sees consistent identities under canonicalization):
    #   0    = alive
    #  -1    = captured by self
    #  1..7  = captured by canonical opponent at that channel
    opp_contacted = [
        np.full((H, W), float(state.opp_contacted[opp]), dtype=np.float32)
        for opp in opp_slots
    ]
    opp_captured_by = []
    for opp in opp_slots:
        captor_raw = int(state.opp_captured_by[opp])
        if captor_raw == -1:
            val = 0.0  # still alive
        elif captor_raw == perspective_slot:
            val = -1.0  # captured by self
        else:
            # Translate raw captor slot to its canonical channel index (1..7).
            # opp_slots is in canonical order, so position+1 gives the channel.
            try:
                val = float(opp_slots.index(captor_raw) + 1)
            except ValueError:
                # Captor not in opp_slots — shouldn't happen unless P!=8.
                val = 0.0
        opp_captured_by.append(np.full((H, W), val, dtype=np.float32))
    return [*opp_contacted, *opp_captured_by]


def _encode_ownership_transition(
    own_newer: np.ndarray,
    own_older: np.ndarray,
    perspective_slot: int,
    opp_slots: list[int],
) -> np.ndarray:
    """Per-cell categorical encoding of an ownership transition (5.05-1 §G).

    Returns a `[H, W]` float32 with:
      - `0.0`  no change (or uncategorized transition, e.g. mountains).
      - `-1.0` self lost (older was perspective; ownership has changed).
      - `+0.5` neutral lost (older was neutral; ownership has changed).
      - `+(1 + k/8)` opp at canonical channel k lost, for k = 1..7.

    Sign separates self-involved (negative) from informational (positive);
    magnitude separates neutral from opponent. Encoder keys on the *older*
    owner only — the gainer's identity isn't part of the categorical signal.
    """
    H, W = own_newer.shape
    out = np.zeros((H, W), dtype=np.float32)
    changed = own_newer != own_older
    out[changed & (own_older == perspective_slot)] = -1.0
    out[changed & (own_older == -1)] = 0.5
    for k, opp in enumerate(opp_slots, start=1):
        out[changed & (own_older == opp)] = 1.0 + k / 8.0
    return out


def _cat_dense_history(
    state: MemoryState,
    perspective_slot: int,
    opp_slots: list[int],
    H: int, W: int,
) -> list[np.ndarray]:
    # Cat 10: Dense recent spatial history (2N channels, N=DENSE_HISTORY_N).
    # ownership_transition[t-k]: categorical encoding keyed on the older
    # owner (see `_encode_ownership_transition`).
    # army_delta[t-k]: signed-log raw army change at tick t-k+1.
    own_transitions = []
    army_deltas = []
    buf_len = len(state.own_buf)  # how many snapshots we have so far
    for k in range(1, DENSE_HISTORY_N + 1):
        # own_buf is right-aligned at t. Snapshot at t-(k-1) = own_buf[-k];
        # snapshot at t-k = own_buf[-k-1]. We need both for a transition.
        idx_newer = buf_len - k
        idx_older = buf_len - k - 1
        if idx_older < 0:
            # not enough history yet (early in the walk)
            own_transitions.append(np.zeros((H, W), dtype=np.float32))
            army_deltas.append(np.zeros((H, W), dtype=np.float32))
            continue
        own_newer = state.own_buf[idx_newer]
        own_older = state.own_buf[idx_older]
        armies_newer = state.armies_buf[idx_newer]
        armies_older = state.armies_buf[idx_older]
        own_transitions.append(
            _encode_ownership_transition(
                own_newer, own_older, perspective_slot, opp_slots,
            )
        )
        # Signed log: sign * log1p(|delta|). Preserves direction at low cost.
        delta = (armies_newer.astype(np.float32) - armies_older.astype(np.float32))
        army_deltas.append(np.sign(delta) * np.log1p(np.abs(delta)).astype(np.float32))
    return [*own_transitions, *army_deltas]


# ---------------------------------------------------------------------------
# build_obs orchestrator
# ---------------------------------------------------------------------------


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
    Build the 96-channel obs tensor for one (game, perspective, tick).

    Channel layout: see `bc.constants.CHANNEL_ORDER`. Each `_cat_*` helper
    returns its slice in CHANNEL_ORDER position; this function stacks them.

    `step_memory` must have already been called for this `(state, t, vis)` —
    `build_obs` is pure-read w.r.t. `state` (the only side effect is lazy
    cache fills inside `bfs.compute_or_get`, which are idempotent).
    """
    own = sim["ownership"][t].reshape(H, W)
    armies = sim["armies"][t].reshape(H, W)
    max_turns = sim["ownership"].shape[0]

    # Cross-cat helper — bool form feeds both cat 3 (as a channel) and cat 5
    # (the BFS passability policy treats fog-structures as impassable).
    structures_in_fog_mask = (
        state.is_structure
        & ~state.known_mountain
        & ~state.known_city
        & ~state.known_general
    )

    channels = [
        *_cat_visibility(vis),
        *_cat_visible_state(vis, own, armies, perspective_slot, opp_slots),
        *_cat_persistent_map(state, structures_in_fog_mask),
        *_cat_memory(state, perspective_slot, opp_slots),
        *_cat_bfs(state, t, perspective_slot, opp_slots, vis, own, armies,
                  structures_in_fog_mask, bfs_cache, H, W),
        *_cat_self_broadcast(state, t, perspective_slot, max_turns, H, W),
        *_cat_opp_broadcast(state, t, opp_slots, H, W),
        *_cat_scoreboard(state, t, opp_slots, H, W),
        *_cat_contact_capture(state, perspective_slot, opp_slots, H, W),
        *_cat_dense_history(state, perspective_slot, opp_slots, H, W),
    ]

    assert len(channels) == OBS_CHANNELS, (
        f"channel count mismatch: built {len(channels)}, "
        f"CHANNEL_ORDER has {OBS_CHANNELS}"
    )

    obs_unpadded = np.stack(channels, axis=0).astype(np.float32)
    obs = np.zeros((OBS_CHANNELS, H_PADDED, W_PADDED), dtype=np.float32)
    obs[:, :H, :W] = obs_unpadded
    return obs
