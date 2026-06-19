"""Per-perspective running memory: `MemoryState` + its lifecycle.

`MemoryState` is initialized at t=0 (`init_memory` / `init_memory_common`),
advanced once per tick by `step_memory`, and read without mutation by
`build_obs`. This module also owns the step-time dense-history encoders
(`_encode_ownership_transition`, `_encode_army_delta`, `_signed_log`): the
dense-history channels are *encoded* here when `step_memory` appends a snapshot
pair, then merely *read* out of the rolling buffers at build time by
`channels._cat_dense_history`. `compute_known_passable` (the cat-5 BFS
passability policy) lives here too, computed off `MemoryState`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from game_types.state_constants import OWN_FOG, OWN_MOUNTAIN
from training.bc.constants import CITY_TRAVERSABILITY_FACTOR
from training.bc.obs.geometry import (
    PerspectiveView,
    _moore_dilate,
    canonical_slot_order,
    make_perspective_view,
)
from training.bc.obs_config import OBS_CONFIG_DEFAULTS, ObsConfig
from training.bc.player_status import PlayerStatusCtx, precompute_player_status
from training.shared.timing import timer


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
    # Live inference: appended per tick by the caller (single owner), so index t
    # is the scoreboard for snapshot t.
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

    # ---- Per-game player status (only when the status group is enabled) ----
    # Drives the `_cat_player_status` channels (is_present / is_alive). Built
    # once in `init_memory` from the full sim (training); the live inference path
    # refreshes it per tick as events arrive. None when the group is disabled.
    player_status: PlayerStatusCtx | None = None

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


def init_memory_common(
    sim: Mapping[str, np.ndarray | list[np.ndarray]],
    perspective_slot: int,
    H: int,
    W: int,
    obs_cfg: ObsConfig,
    P: int = 8,
) -> MemoryState:
    """Per-game memory scaffolding with empty scoreboard history — the shared
    base of both memory-init paths.

    The live path (BC inference / eval bot) calls this directly and is the single
    owner of per-tick history: it appends one row per tick via `scoreboard_row` +
    `step_memory`, so `*_count_history[t]` is the scoreboard for snapshot t. The
    training path (`init_memory`) extends this by precomputing all rows at once.
    """
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


# ============================================================================
# TODO(obs-alive-tick-seam): board-snapshot-time. This army channel reads
# ownership[t] — the PRE-RESOLUTION board. At a capture-while-alive tick the
# victim's tiles haven't transferred yet (they zero at t+1), so obs shows them
# on-board with army, while alive_mask (bc/targets/elim_targets.py) uses
# death>t (event-time) and reads them DEAD at the same frame t. The pair
# describes opposite ends of the tick -> a 1-frame obs/target inconsistency
# (~1% of frames). Paired site: alive_mask. Deferred to the obs/head rework.
# See docs/2026-06/6.18-6-obs-alive-tick-seam.md.
# ============================================================================
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
    """Training variant: init_memory_common + precompute the scoreboard for all
    T rows off the full [T, HW] array (vectorized — the live path can't, it gets
    rows one tick at a time)."""
    state = init_memory_common(sim, perspective_slot, H, W, obs_cfg, P)

    ownership = sim["ownership"]  # [T, HW] int8
    armies = sim["armies"]  # [T, HW] int16
    T = ownership.shape[0]
    land_buf = np.zeros((T, P), dtype=np.int32)
    army_buf = np.zeros((T, P), dtype=np.int64)
    for p in range(P):
        owned_mask = (ownership == p)  # [T, HW]
        land_buf[:, p] = owned_mask.sum(axis=1)
        # TODO(obs-alive-tick-seam): board-snapshot-time army (see scoreboard_row
        # banner / docs/2026-06/6.18-6-obs-alive-tick-seam.md).
        army_buf[:, p] = (armies * owned_mask).sum(axis=1)
    state.land_count_history = [land_buf[t] for t in range(T)]
    state.army_count_history = [army_buf[t] for t in range(T)]

    # Player-status precompute (once per game) for the is_present / is_alive
    # channels. The full training sim carries every death/capture/neutralize
    # event, so a single precompute is correct for every frame.
    if obs_cfg.player_status_channels:
        state.player_status = precompute_player_status(sim)

    return state


def init_memory_live_fog_only(
    sim: Mapping[str, np.ndarray | list[np.ndarray]],
    perspective_slot: int,
    H: int,
    W: int,
    P: int = 8,
) -> MemoryState:
    """`init_memory_common` for consumers that track fog memory but never encode
    the dense-history obs channels (e.g. the heuristic eval bot's `WorldModel`).
    The obs-encoder config is immaterial to them, so this fills the default —
    they read `MemoryState`'s fog/scoreboard fields, never the dense-history
    buffers. NN inference must call `init_memory_common` with the checkpoint's
    own `obs_cfg` instead, so its obs matches the model's `in_ch`.
    """
    return init_memory_common(sim, perspective_slot, H, W, OBS_CONFIG_DEFAULTS, P)


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
