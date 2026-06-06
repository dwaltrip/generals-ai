"""
Observation-tensor construction — the policy/value network's input at one
(game, perspective, timestep) snapshot.

This module owns three things:
  - `canonical_slot_order` — pin from Phase 1 (perspective→slot-0 layout).
  - `MemoryState` + `init_memory` + `step_memory` — per-perspective running
    memory, advanced once per tick.
  - `build_obs` — pure read of (sim, state, vis, bfs_cache) →
    [obs_channel_count(n), H_PADDED, W_PADDED].

The channel count is `obs_channel_count(state.obs_cfg.dense_history_n)`: a fixed
base block plus a `2 * n` dense-history tail. The obs-encoder config rides on
`MemoryState` (set at `init_memory`), so `build_obs` reads `n` off the state it
already receives rather than taking another parameter.

The channel set is grouped by category internally (named vars per channel),
then assembled at the bottom of `build_obs`; that stacking order is the source
of truth for layout (the names in `bc.constants` are reference docs). The
named-var shape is deliberately verbose so that the encoding for each channel is
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
  - opp_N_city_inference (currently zero-stubbed — full encoding is the
    "infer cities from peacetime growth" heuristic from 5.05-1 §F).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from bc import bfs
from bc.constants import (
    CITY_TRAVERSABILITY_FACTOR,
    H_PADDED,
    W_PADDED,
    obs_channel_count,
)
from bc.obs_config import OBS_CONFIG_DEFAULTS, ObsConfig
from game_types.state_constants import OWN_FOG, OWN_MOUNTAIN
from shared.timing import timer


# ---------------------------------------------------------------------------
# Canonicalization (unchanged from Phase 1)
# ---------------------------------------------------------------------------


def _moore_dilate(mask: np.ndarray) -> np.ndarray:
    """Moore-neighborhood (3x3) binary dilation of a 2D bool mask.

    Same 8-direction OR pattern as `visibility.compute_visibility`, used by
    `step_memory` when expanding the agent's vision of an opponent's tile
    into "opponent had vision around here."
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


def pad_initial_generals(
    initial_generals, perspective_slot: int, num_slots: int = 8,
) -> np.ndarray:
    """Pad `initial_generals` to `num_slots` length for the obs encoder.

    The BC encoder indexes slots 0..num_slots-1 unconditionally. For
    <num_slots-player games, unused slots are filled with the perspective's own
    general cell — idempotent under fancy indexing (sets an already-set
    own-general bit) and safe in step_memory's per-player loop (the own general
    is known from t=0, so padded slots never trigger false sightings).
    """
    ig = np.asarray(initial_generals, dtype=np.int32)
    if len(ig) >= num_slots:
        return ig
    own = int(ig[perspective_slot])
    padded = np.full(num_slots, own, dtype=np.int32)
    padded[: len(ig)] = ig
    return padded


# ---------------------------------------------------------------------------
# Perspective-filtered board view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerspectiveView:
    """A single-tick board state filtered to what the perspective observed.

    Cells outside the perspective's vision at this tick have `own = OWN_FOG`
    and `armies = 0` — matching what the perspective would see in the live
    game under fog of war. Cells within vision carry their true sim values.

    Used by `step_memory` to populate the dense-history buffer; the same
    abstraction is available for any future consumer that needs a per-tick
    "what the perspective sees" snapshot (e.g., a future pre-filter at the
    top of `build_obs` for current-tick cats).
    """
    own: np.ndarray    # int8 [H, W]
    armies: np.ndarray  # int16 [H, W]


def make_perspective_view(
    own_raw: np.ndarray, armies_raw: np.ndarray, vis: np.ndarray,
) -> PerspectiveView:
    """Build a `PerspectiveView` from raw sim arrays + the perspective's
    current vision mask. Fog cells get the OWN_FOG sentinel and zero armies.
    """
    return PerspectiveView(
        own=np.where(vis, own_raw, OWN_FOG).astype(np.int8),
        armies=np.where(vis, armies_raw, 0).astype(np.int16),
    )


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
      - Static per game: `is_structure`.
      - Scoreboard: `land_count_history`, `army_count_history`. Training:
        precomputed for all T rows. Live: appended per tick.
      - Per-cell mutable masks: `historically_seen`, `known_mountain`,
        `known_city`, `known_general`, plus the `last_seen_*` channels and
        `turns_since_seen`.
      - Per-opponent: `opp_contacted` (binary monotonic), `opp_captured_by`
        (-1=alive, 0..7=raw slot of captor).
      - Source pins: `general_locations` (-1 = unknown).
      - Dense-history: `prev_view` (one snapshot, used to diff against the
        next tick's snapshot inside `step_memory`) plus `transition_buf` /
        `army_delta_buf` — bounded deques of already-encoded `[H, W] float32`
        channels, length `obs_cfg.dense_history_n`. `_cat_dense_history`
        reads these directly.
      - Static-per-walk: `obs_cfg`, `perspective_slot`, `opp_slots`. The
        per-opp channel groupings index `opp_slots` consistently across
        every cat that emits them.
    """

    # ---- Static-per-walk identifiers ----
    # Encoder hyperparameters (currently `dense_history_n`); sizes the
    # transition/army-delta buffers and the dense-history channel tail.
    obs_cfg: ObsConfig
    # Raw slot id of the perspective player and the canonical 7 opponents.
    # `opp_slots = canonical_slot_order(perspective_slot)[1:]`; pinned at init
    # so per-frame helpers don't recompute it.
    perspective_slot: int
    opp_slots: list[int]

    # ---- Static within a game ----
    # bool [H, W]: cells the agent knows are *some* structure (mountain, city,
    # or general) — visible terrain at t=0. Per game-mechanics, structure
    # presence is visible through fog; only the *type* is hidden.
    is_structure: np.ndarray
    # int32 [P]: flat (unpadded) cell index of each player's general; -1 if
    # the perspective hasn't observed it. Updated by step_memory on first
    # sighting. Self general is set in init_memory (always known at t=0).
    general_locations: np.ndarray
    # list of (P,) int32 arrays: per-tick land count (cells owned).
    # Training: precomputed for all T rows in init_memory.
    # Live inference: row 0 from init_memory_live, appended per tick.
    land_count_history: list[np.ndarray]
    # list of (P,) int64 arrays: per-tick total army. Same shape/lifecycle.
    army_count_history: list[np.ndarray]

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
    # -----------------------------------------------------------------
    # TODO: The magic values of -1, -2 for `last_seen_owner` are not well documented.
    #   Should be constants, I believe.
    #   Also, they differ in meaning. -1 means "never seen" elsewhere.
    # TODO: -1 should also be a constant for `last_seen_armies`, etc.
    # -----------------------------------------------------------------
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

    # ---- Dense-history caches (advanced by step_memory) ----
    # Previous tick's perspective-filtered snapshot. Held so `step_memory`
    # can diff (own_t, armies_t) against (own_t-1, armies_t-1).
    prev_view: PerspectiveView | None = None
    # Encoded ownership-transition channels, oldest-first. float32 [H, W].
    # maxlen = obs_cfg.dense_history_n.
    transition_buf: deque = field(default_factory=deque)
    # Production-subtracted, signed-log army-delta channels, paired 1:1
    # with `transition_buf`. float32 [H, W].
    army_delta_buf: deque = field(default_factory=deque)


def _init_memory_common(
    sim: Mapping[str, np.ndarray | list[np.ndarray]],
    perspective_slot: int,
    H: int,
    W: int,
    P: int,
    obs_cfg: ObsConfig,
) -> MemoryState:
    """Shared setup for init_memory / init_memory_live. Returns a
    MemoryState with empty scoreboard lists (caller fills them)."""
    HW = H * W

    # The static "is this cell a structure" mask: mountains + initial cities
    # + the perspective's own general. Per game-mechanics §5, mountain and
    # city existence is always visible through fog, so those positions are
    # known from t=0. Opp generals are deliberately excluded — per §5, enemy
    # generals behind fog appear as ordinary empty tiles, not as structures.
    # Post-capture cities (general → city transitions) enter the visible-
    # structure set via cities_present_at; that's folded into the fog-
    # structure mask dynamically in build_obs rather than mutating this set.
    is_structure_flat = np.zeros(HW, dtype=bool)
    is_structure_flat[sim["mountains"]] = True
    is_structure_flat[sim["initial_cities"]] = True
    is_structure_flat[int(sim["initial_generals"][perspective_slot])] = True
    is_structure = is_structure_flat.reshape(H, W)

    # Self general known at t=0; opponents unknown.
    general_locations = np.full(P, -1, dtype=np.int32)
    general_locations[perspective_slot] = int(sim["initial_generals"][perspective_slot])

    state = MemoryState(
        obs_cfg=obs_cfg,
        perspective_slot=perspective_slot,
        opp_slots=canonical_slot_order(perspective_slot, P)[1:],
        is_structure=is_structure,
        general_locations=general_locations,
        land_count_history=[],
        army_count_history=[],
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
        transition_buf=deque(maxlen=obs_cfg.dense_history_n),
        army_delta_buf=deque(maxlen=obs_cfg.dense_history_n),
    )

    # Mark own general as known from t=0.
    own_loc = int(general_locations[perspective_slot])
    own_r, own_c = divmod(own_loc, W)
    state.known_general[own_r, own_c] = True

    return state


def scoreboard_row(ownership_row: np.ndarray, armies_row: np.ndarray, P: int):
    """Compute (land_counts, army_counts) for a single tick."""
    land = np.zeros(P, dtype=np.int32)
    army = np.zeros(P, dtype=np.int64)
    for p in range(P):
        mask = ownership_row == p
        land[p] = mask.sum()
        army[p] = (armies_row * mask).sum()
    return land, army


def init_memory(
    sim: dict[str, np.ndarray],
    perspective_slot: int,
    H: int,
    W: int,
    obs_cfg: ObsConfig,
    P: int = 8,
) -> MemoryState:
    """Training path: precompute scoreboard for all T rows."""
    state = _init_memory_common(sim, perspective_slot, H, W, P, obs_cfg)

    ownership = sim["ownership"]  # [T, HW] int8
    armies = sim["armies"]  # [T, HW] int16
    T = ownership.shape[0]
    land_buf = np.zeros((T, P), dtype=np.int32)
    army_buf = np.zeros((T, P), dtype=np.int64)
    for p in range(P):
        owned_mask = (ownership == p)  # [T, HW]
        land_buf[:, p] = owned_mask.sum(axis=1)
        army_buf[:, p] = (armies * owned_mask).sum(axis=1)
    state.land_count_history = [land_buf[t] for t in range(T)]
    state.army_count_history = [army_buf[t] for t in range(T)]

    return state


def init_memory_live(
    sim: Mapping[str, np.ndarray | list[np.ndarray]],
    perspective_slot: int,
    H: int,
    W: int,
    obs_cfg: ObsConfig,
    P: int = 8,
) -> MemoryState:
    """Live inference path: compute scoreboard for row 0 only.
    Caller appends subsequent rows each tick."""
    state = _init_memory_common(sim, perspective_slot, H, W, P, obs_cfg)

    land_0, army_0 = scoreboard_row(sim["ownership"][0], sim["armies"][0], P)
    state.land_count_history = [land_0]
    state.army_count_history = [army_0]

    return state


def init_memory_live_fog_only(
    sim: Mapping[str, np.ndarray | list[np.ndarray]],
    perspective_slot: int,
    H: int,
    W: int,
    P: int = 8,
) -> MemoryState:
    """Live `init_memory_live` for consumers that track fog memory but never
    encode the dense-history obs channels (e.g. the heuristic eval bot's
    `WorldModel`). The obs-encoder config is immaterial to them, so this fills
    the default — they read `MemoryState`'s fog/scoreboard fields, never the
    dense-history buffers. NN inference must use `init_memory_live` with the
    checkpoint's own `obs_cfg` instead, so its obs matches the model's `in_ch`.
    """
    return init_memory_live(sim, perspective_slot, H, W, OBS_CONFIG_DEFAULTS, P)


@timer.timed("step_memory")
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

    # Add newly discovered mountains to `known_mountain` explored in prev tick)
    new_mountain = vis & (own_t == OWN_MOUNTAIN) & ~state.known_mountain
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

    # Dense-history: build the current tick's perspective-filtered snapshot
    # (fog cells get OWN_FOG / 0 so the encoded channels stay fog-respecting
    # per game-mechanics §5), then — if there's a previous snapshot to pair
    # with — diff against it once and push the encoded transition + army
    # delta into the rolling buffers `_cat_dense_history` reads from.
    new_view = make_perspective_view(own_t, armies_t, vis)
    if state.prev_view is not None:
        _append_dense_history_pair(state, sim, t, new_view, H, W)
    state.prev_view = new_view

    return graph_grew


def _append_dense_history_pair(
    state: MemoryState,
    sim: dict[str, np.ndarray],
    t: int,
    new_view: PerspectiveView,
    H: int,
    W: int,
) -> None:
    """Encode the (new_view, prev_view) snapshot pair into transition +
    army-delta channels; append to the rolling buffers. Caller guarantees
    `state.prev_view is not None`.
    """
    prev_view = state.prev_view
    assert prev_view is not None  # for type-narrowing; caller-guaranteed
    own_newer, own_older = new_view.own, prev_view.own
    armies_newer, armies_older = new_view.armies, prev_view.armies

    # both_observed: cells the perspective saw at both endpoints. Anything
    # outside this mask emits 0 in both channels (5.05-1 §7.2).
    both_observed = (
        (own_newer != OWN_FOG) & (own_older != OWN_FOG)
    ).astype(np.float32)

    transition = _encode_ownership_transition(
        own_newer, own_older, state.perspective_slot, state.opp_slots,
    )

    # City/general masks at t — same derivation as the `cities_at_t` block in
    # step_memory's known_city update. Feeds the production-subtraction in
    # `_encode_army_delta`.
    HW = H * W
    cities_at_t_flat = sim["cities"][sim["cities_present_at"] <= t]
    city_mask_flat = np.zeros(HW, dtype=bool)
    city_mask_flat[cities_at_t_flat] = True
    initial_generals_flat = np.zeros(HW, dtype=bool)
    initial_generals_flat[sim["initial_generals"]] = True
    city_mask = city_mask_flat.reshape(H, W)
    general_mask = (initial_generals_flat & ~city_mask_flat).reshape(H, W)

    delta = _signed_log(_encode_army_delta(
        armies_newer, armies_older, own_newer, t,
        city_mask, general_mask,
    ))

    state.transition_buf.append(transition * both_observed)
    state.army_delta_buf.append(delta * both_observed)


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
    total_army = int(state.army_count_history[t][perspective_slot])

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
# `[H, W] float32` channels in stack-order position. `build_obs` is the
# orchestrator that calls them and stacks the result. Comments retain the
# numeric category labels (Cat 1, Cat 2, ...) for cross-reference with the
# design docs (5.05-1 §F/§G, etc.) and the channel groupings in `bc.constants`.
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
    # Cat 5: BFS distance-from-known-generals (8 channels). `maybe_invalidate`
    # gates the per-source BFS on mask change so structurally-stable frames
    # are cache hits.
    known_passable_flat = compute_known_passable(
        state, t, perspective_slot, vis, own, armies,
        structures_in_fog_mask, H, W,
    )
    bfs_cache.maybe_invalidate(known_passable_flat)
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


_TIMESTEP_DIVISOR = 1000

def _cat_self_broadcast(
    state: MemoryState,
    t: int,
    perspective_slot: int,
    H: int,
    W: int,
) -> list[np.ndarray]:
    # Cat 6: Self broadcast scalars (3 channels). Same value at every cell
    # of the unpadded board; padding stays zero.
    self_army_count = np.full(
        (H, W),
        state.army_count_history[t][perspective_slot] / _ARMY_DIVISOR,
        dtype=np.float32,
    )
    self_land_count = np.full(
        (H, W),
        state.land_count_history[t][perspective_slot] / _LAND_DIVISOR,
        dtype=np.float32,
    )
    timestep = np.full((H, W), t / _TIMESTEP_DIVISOR, dtype=np.float32)
    return [self_army_count, self_land_count, timestep]


def _cat_opp_broadcast(
    state: MemoryState, t: int, opp_slots: list[int], H: int, W: int,
) -> list[np.ndarray]:
    # Cat 7: Per-opponent broadcast scalars (14 channels).
    opp_army_count = [
        np.full((H, W), state.army_count_history[t][opp] / _ARMY_DIVISOR, dtype=np.float32)
        for opp in opp_slots
    ]
    opp_land_count = [
        np.full((H, W), state.land_count_history[t][opp] / _LAND_DIVISOR, dtype=np.float32)
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

    # TODO: 5.05-1 §F encoding
    opp_city_inference = [np.zeros((H, W), dtype=np.float32) for _ in opp_slots]
    t_prev = max(0, t - _LAND_DELTA_WINDOW)
    opp_land_delta = [
        np.full(
            (H, W),
            (state.land_count_history[t][opp] - state.land_count_history[t_prev][opp])
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


def _encode_army_delta(
    armies_newer: np.ndarray,
    armies_older: np.ndarray,
    own_newer: np.ndarray,
    t_newer: int,
    city_mask: np.ndarray,
    general_mask: np.ndarray,
) -> np.ndarray:
    """Production-subtracted raw army delta (5.05-1 §G + §7.2). Caller
    applies `_signed_log` for the channel encoding.

    Subtracts the per-cell expected production applied between
    `snapshot[t_newer-1]` and `snapshot[t_newer]`. Production rules
    (sim-core README — production fires post-step-increment):
      - At `t_newer % 2 == 0`: each owned general or city gains +1 army.
      - At `t_newer % 50 == 0` (land tick): each owned cell gains +1 army.

    Subtraction is universal: at combat cells the +1/+2 correction is
    dwarfed by the combat delta, so the signal stays dominated by combat
    where it matters. The goal is to strip peacetime production noise so
    the channel highlights real events.

    Returns float32 [H, W]: raw signed adjusted delta.

    TODO: hand-coded fixtures — small boards, picked ticks exercising the
    production cases (t%2==0 only, t%50==0 only, both, neither), assert
    adjusted-delta values cell-by-cell.
    """
    is_owned = own_newer >= 0
    prod = np.zeros_like(armies_newer, dtype=np.float32)
    if t_newer % 2 == 0:
        prod += ((city_mask | general_mask) & is_owned).astype(np.float32)
    if t_newer % 50 == 0:
        prod += is_owned.astype(np.float32)
    raw_delta = armies_newer.astype(np.float32) - armies_older.astype(np.float32)
    return raw_delta - prod


def _signed_log(x: np.ndarray) -> np.ndarray:
    """`sign(x) * log1p(|x|)` — soft compression for signed magnitudes.
    Identity at 0; ≈ log magnitude for large |x|. Used for the army-delta
    channel.
    """
    return (np.sign(x) * np.log1p(np.abs(x))).astype(np.float32)


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
    state: MemoryState, H: int, W: int,
) -> list[np.ndarray]:
    # Cat 10: Dense recent spatial history (2N channels, N=obs_cfg.dense_history_n).
    # Each pair is encoded once by `step_memory` when the snapshot at its
    # newer endpoint is appended; we just read out of the rolling buffers
    # here.
    #
    # `transition_buf` / `army_delta_buf` are right-aligned at the current
    # tick: index [-1] is the (t, t-1) pair, [-2] is (t-1, t-2), etc. Early
    # in a walk the buffers haven't filled yet, so a missing slot emits a
    # zero channel.
    n = state.obs_cfg.dense_history_n
    buf_len = len(state.transition_buf)
    zero = np.zeros((H, W), dtype=np.float32)
    own_transitions = [
        state.transition_buf[buf_len - k] if buf_len - k >= 0 else zero
        for k in range(1, n + 1)
    ]
    army_deltas = [
        state.army_delta_buf[buf_len - k] if buf_len - k >= 0 else zero
        for k in range(1, n + 1)
    ]
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
    # buffer. Equivalent to `np.stack(channels).astype(float32)` followed by a
    # pad copy, but in a single write pass instead of three (stack gather +
    # astype copy + pad copy) — the assembly's per-sample memory traffic is the
    # cost that scales with channel count, so collapsing it matters most for
    # large dense-history depths. The pad region stays zero; the slice
    # assignment converts any non-float32 channel exactly as `astype` did.
    with timer.section("assemble"):
        obs = np.zeros((n_channels, H_PADDED, W_PADDED), dtype=np.float32)
        for i, ch in enumerate(channels):
            obs[i, :H, :W] = ch
    return obs
