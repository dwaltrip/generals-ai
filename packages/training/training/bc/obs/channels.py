"""Channel-category builders — the obs tensor's content, one `_cat_*` per group.

Each `_cat_*` builds one category of the obs tensor and returns a list of
`[H, W] float32` channels in stack-order position. `build.build_obs` is the
orchestrator that calls them and stacks the result. Comments retain the numeric
category labels (Cat 1, Cat 2, ...) for cross-reference with the design docs
(5.05-1 §F/§G, etc.) and the channel groupings in `bc.constants`.

`structures_in_fog_mask` is the one cross-cat helper — it's the bool form of
cat 3's `structures_in_fog` channel and also feeds the BFS passability policy
in cat 5. Computed once in `build_obs` and threaded into both.

--- Per-opp slot ordering contract ---

Every per-opp channel group (`opp_N_owned`, `opp_N_army_count`,
`last_seen_owner_opp_N`, BFS-to-opp-N, `opp_N_contacted`, `opp_N_captured_by`,
`opp_N_has_seen`, ...) follows the same canonical mapping: channel index `i`
for `i ∈ 1..7` corresponds to the raw slot `opp_slots[i-1]`, where
`opp_slots = canonical_slot_order(perspective)[1:]`. Channel 0 (in any per-opp
grouping that *includes* self) is always the perspective player.

--- Knobs deferred for later tuning ---

  - BFS-policy knobs (cat 5): see the cat-5 comment for the spike's
    enemy-general / city-passability decisions, and `bfs.py`'s knob header for
    the weighted-edge upgrade path.
  - Broadcast-scalar normalization divisors (currently: hand-picked typical
    late-game values; could be game-normalized or rolling-window scaled).
  - opp_N_city_inference (currently zero-stubbed — full encoding is the
    "infer cities from peacetime growth" heuristic from 5.05-1 §F).
"""

from __future__ import annotations

import numpy as np

from training.bc import bfs
from training.bc.obs.memory import MemoryState, compute_known_passable


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
