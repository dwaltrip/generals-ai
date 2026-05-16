"""Shared library for sim integration fixture seed + regen + test.

Fixture layout:
    tests/fixtures/<name>/
        wire.bin      - raw replay wire_data blob (input)
        expected.npz  - oracle (snapshots + end-state + events + damage);
                        full schema documented on `pack_sim_output` below

Manifest below lists each fixture's replay_id + scenario tag.
  - `seed_fixtures.py`  pulls wire.bin from the collector DB (one-time).
  - `regen_fixtures.py` runs the sim and writes expected.npz.
  - `test_sim_integration.py` re-runs the sim and compares.

Snapshot timing: see `sim-core/README.md`. Snapshot windows around
events follow the "pre / on / after" convention:
    Land tick at K (production):       [K-1, K, K+1]
    Capture/death/neutralize at e:     [e, e+1, e+2]   (effect at e+1)
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FIXTURES_DIR = Path(__file__).parent / "fixtures"

# 3-way action spatial/time bucketing (mirrors find_fixture_candidates.py).
REGION = 5
T_BUCKET = 100

# 2-on-1 general: how far back to scan moves for general-tile attackers.
GEN_ATTACK_WINDOW = 3

# 3-way action: minimum gap between picked action ticks (avoid overlapping
# triplets when consecutive timesteps are both high-action).
MIN_3WAY_TICK_GAP = 3


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    replay_id: str
    scenario: str


# Initial picks from the candidate-finder sweep (scripts/find_fixture_candidates.py
# run at per-bucket=50). Replace any of these after manual viewer-review pass.
FIXTURES: list[FixtureSpec] = [
    FixtureSpec("3way_action_a",             "wccUXL95D", "3way_action"),
    FixtureSpec("3way_action_b",             "1gUvzdQk0", "3way_action"),
    FixtureSpec("2on1_general_a",            "zN4zX1ChP", "2on1_general"),
    FixtureSpec("2on1_general_b",            "y3AZo1KQS", "2on1_general"),
    FixtureSpec("capture_during_surrender_a", "V3ylC7DCS", "capture_during_surrender"),
    FixtureSpec("capture_during_surrender_b", "wKZmzOuc9", "capture_during_surrender"),
    FixtureSpec("samet_chain_a",             "wbl7gM7_8", "samet_chain"),
    FixtureSpec("samet_chain_b",             "f0tFZV3bs", "samet_chain"),
    FixtureSpec("multi_surrender_a",         "V66AtVZC6", "multi_surrender"),
    FixtureSpec("multi_surrender_b",         "rzzNOi9fD", "multi_surrender"),
    FixtureSpec("city_battling_a",           "ZAndNfDLq", "city_battling"),
    FixtureSpec("city_battling_b",           "aUZbeAiz7", "city_battling"),
    FixtureSpec("run_of_the_mill_a",         "2jH6nzK-_", "run_of_the_mill"),
    FixtureSpec("run_of_the_mill_b",         "6kLys2cKt", "run_of_the_mill"),
    FixtureSpec("short_game_a",              "ab63tPA6c", "short_game"),
    FixtureSpec("short_game_b",              "MyNuwrc4g", "short_game"),
]


# ============================================================================
# Snapshot timestep selection
# ============================================================================

def fixture_snap_indices(state, replay, scenario: str) -> list[int]:
    """Return sorted, deduped snapshot indices for a fixture. Combines the
    standard set (applied to every fixture) with per-scenario extras."""
    own_stack = np.stack(state.snapshots_ownership, axis=0)
    T = own_stack.shape[0]
    ts = _standard_timesteps(state, T)
    ts |= _scenario_timesteps(state, replay, scenario, own_stack)
    return sorted(t for t in ts if 0 <= t < T)


def _standard_timesteps(state, T: int) -> set[int]:
    """Timesteps captured for every fixture, regardless of scenario."""
    ts: set[int] = set()

    # Land tick triplets — production fires post-increment, effect at snap[K]
    for K in (50, 150):
        for t in (K - 1, K, K + 1):
            ts.add(t)

    # First kill — death event fires pre-increment, effect at snap[e+1]
    if state.death_events:
        e = state.death_events[0].timestep
        for t in (e, e + 1, e + 2):
            ts.add(t)

    # Early absolute
    for t in (25, 40, 70):
        ts.add(t)

    # Mid-game absolute
    for t in (120, 170):
        ts.add(t)

    # Near-end percentiles
    for pct in (0.85, 0.90, 0.95):
        ts.add(int(T * pct))

    # Final 3
    for t in (T - 3, T - 2, T - 1):
        ts.add(t)

    return {t for t in ts if 0 <= t < T}


def _scenario_timesteps(state, replay, scenario: str, own_stack: np.ndarray) -> set[int]:
    fn = _SCENARIO_FNS.get(scenario)
    if fn is None:
        return set()
    return fn(state, replay, own_stack)


# ============================================================================
# Per-scenario interesting-timestep finders
#
# Each returns a set of snapshot indices to capture. The orchestrator
# unions these with the standard set, clamps to [0, T-1], dedupes.
# ============================================================================


def _ts_2on1_general(state, replay, own_stack) -> set[int]:
    """5-tick window [e-1..e+3] around the capture with the most distinct
    attackers targeting the captured player's general tile in the window
    [e - GEN_ATTACK_WINDOW, e]."""
    moves = replay.moves
    if len(moves.timestep) == 0:
        return set()
    initial_generals = list(replay.static.initial_generals)
    best_e = None
    best_count = 0
    for ce in state.capture_events:
        gt = initial_generals[ce.captured]
        if gt < 0:
            continue
        i0 = int(np.searchsorted(moves.timestep, ce.timestep - GEN_ATTACK_WINDOW, side="left"))
        i1 = int(np.searchsorted(moves.timestep, ce.timestep, side="right"))
        if i1 <= i0:
            continue
        mask = (moves.dest[i0:i1] == gt) & (moves.index[i0:i1] != ce.captured)
        attackers = set(int(x) for x in moves.index[i0:i1][mask].tolist())
        if len(attackers) > best_count:
            best_count = len(attackers)
            best_e = ce.timestep
    if best_e is None:
        return set()
    return {best_e - 1, best_e, best_e + 1, best_e + 2, best_e + 3}


def _ts_samet_chain(state, replay, own_stack) -> set[int]:
    """5-tick window around the first timestep where captured(e1) == captor(e2)
    among same-timestep capture events (A->B->C chain)."""
    if not state.capture_events:
        return set()
    by_t: dict[int, list] = {}
    for ce in state.capture_events:
        by_t.setdefault(ce.timestep, []).append(ce)
    for t in sorted(by_t):
        events = by_t[t]
        if len(events) < 2:
            continue
        captors = {ce.captor for ce in events}
        captured = {ce.captured for ce in events}
        if captors & captured:
            return {t - 1, t, t + 1, t + 2, t + 3}
    return set()


def _ts_capture_during_surrender(state, replay, own_stack) -> set[int]:
    """5-tick window around the first capture event for a player whose first
    AFK entry precedes the capture (surrendered but not yet neutralized)."""
    afks = replay.afks
    if len(afks.timestep) == 0:
        return set()
    first_afk: dict[int, int] = {}
    for idx, t in zip(afks.index.tolist(), afks.timestep.tolist()):
        first_afk.setdefault(int(idx), int(t))
    for ce in state.capture_events:
        afk_t = first_afk.get(ce.captured)
        if afk_t is not None and afk_t < ce.timestep:
            e = ce.timestep
            return {e - 1, e, e + 1, e + 2, e + 3}
    return set()


def _ts_city_battling(state, replay, own_stack) -> set[int]:
    """For the top 3 city tiles by total ownership transitions, take a triplet
    around the middle flip event on that tile."""
    cities = state.cities
    if not cities:
        return set()
    cities_arr = np.array(cities, dtype=np.int64)
    transitions_per_tile = (own_stack[1:] != own_stack[:-1]).sum(axis=0)
    city_trans = transitions_per_tile[cities_arr]
    top3 = np.argsort(-city_trans)[:3]

    ts: set[int] = set()
    for k in top3:
        if city_trans[k] == 0:
            continue
        tile = int(cities_arr[k])
        # Indices where own_stack[i+1, tile] != own_stack[i, tile]; the flip
        # event timestep is i (pre-increment).
        diffs = np.where(own_stack[1:, tile] != own_stack[:-1, tile])[0]
        if len(diffs) == 0:
            continue
        f = int(diffs[len(diffs) // 2])
        ts.update((f, f + 1, f + 2))
    return ts


def _ts_3way_action(state, replay, own_stack) -> set[int]:
    """Within the peak (region, time-bucket), pick up to 4 high-action ticks
    (most ownership changes within the region between consecutive snapshots),
    each as a triplet [a, a+1, a+2]. Min-gap enforced to avoid overlap."""
    s = replay.static
    H, W = s.map_height, s.map_width
    T = own_stack.shape[0]
    n_ry, n_rx = H // REGION, W // REGION
    if n_ry == 0 or n_rx == 0:
        return set()
    Hp, Wp = n_ry * REGION, n_rx * REGION
    n_buckets = (T + T_BUCKET - 1) // T_BUCKET
    Tp = n_buckets * T_BUCKET

    # Per-player presence in each (time-bucket, region) cell.
    own_2d = own_stack.reshape(T, H, W)
    presence = np.zeros((state.num_players, n_buckets, n_ry, n_rx), dtype=bool)
    for p in range(state.num_players):
        pm = (
            (own_2d[:, :Hp, :Wp] == p)
            .reshape(T, n_ry, REGION, n_rx, REGION)
            .any(axis=(2, 4))
        )
        if T < Tp:
            pad = np.zeros((Tp - T, n_ry, n_rx), dtype=bool)
            pm = np.concatenate([pm, pad], axis=0)
        pm = pm.reshape(n_buckets, T_BUCKET, n_ry, n_rx).any(axis=1)
        presence[p] = pm

    distinct = presence.sum(axis=0)
    if distinct.max() < 3:
        return set()
    B, ry, rx = np.unravel_index(int(distinct.argmax()), distinct.shape)

    # Tile indices for the peak 5x5 region.
    region_tiles = [
        (int(ry) * REGION + dy) * W + (int(rx) * REGION + dx)
        for dy in range(REGION) for dx in range(REGION)
    ]
    region_tiles_arr = np.array(region_tiles, dtype=np.int64)

    # Restrict to peak time-bucket; rank snapshot indices by per-step
    # ownership-change count within the region.
    t_lo = int(B) * T_BUCKET
    t_hi = min((int(B) + 1) * T_BUCKET, T)
    region_own = own_stack[t_lo:t_hi, region_tiles_arr]
    if region_own.shape[0] < 2:
        return set()
    changes = (region_own[1:] != region_own[:-1]).sum(axis=1)

    # Greedy pick top action-ticks with min-gap to avoid overlapping triplets.
    chosen: list[int] = []
    for idx in np.argsort(-changes):
        if changes[idx] == 0:
            break
        a = t_lo + int(idx)
        if any(abs(a - c) < MIN_3WAY_TICK_GAP for c in chosen):
            continue
        chosen.append(a)
        if len(chosen) == 4:
            break

    ts: set[int] = set()
    for a in chosen:
        ts.update((a, a + 1, a + 2))
    return ts


def _ts_multi_surrender(state, replay, own_stack) -> set[int]:
    """For the top 2 surrendered players (by AFK-entry count), take a triplet
    around their first AFK event (surrender) and a triplet around their second
    AFK event (neutralize), if present."""
    afks = replay.afks
    if len(afks.timestep) == 0:
        return set()
    by_player: dict[int, list[int]] = {}
    for idx, t in zip(afks.index.tolist(), afks.timestep.tolist()):
        by_player.setdefault(int(idx), []).append(int(t))
    # Top by event count, then by earliest surrender for stability.
    top2 = sorted(by_player.items(), key=lambda kv: (-len(kv[1]), kv[1][0]))[:2]

    ts: set[int] = set()
    for _player, events in top2:
        events_sorted = sorted(events)
        s = events_sorted[0]
        ts.update((s, s + 1, s + 2))
        if len(events_sorted) >= 2:
            n = events_sorted[1]
            ts.update((n, n + 1, n + 2))
    return ts


_SCENARIO_FNS = {
    "2on1_general":              _ts_2on1_general,
    "samet_chain":               _ts_samet_chain,
    "capture_during_surrender":  _ts_capture_during_surrender,
    "city_battling":             _ts_city_battling,
    "3way_action":               _ts_3way_action,
    "multi_surrender":           _ts_multi_surrender,
    # run_of_the_mill, short_game: no scenario extras
}


# ============================================================================
# Oracle packaging — shared between regen (write) and test (compare).
# ============================================================================

def pack_sim_output(state, replay, snap_indices: list[int]) -> dict[str, np.ndarray]:
    """
    Build the dict written to expected.npz and compared against in the test.

    Schema. Dimension symbols:
      N           = len(snap_indices)
      map_size    = map_width * map_height
      P           = state.num_players
      C           = len(state.cities) at game end
      n_captures, n_neutralizes, n_deaths = event-list lengths

        snap_indices         int32   (N,)               sorted snapshot indices
        snaps_ownership      int8    (N, map_size)
        snaps_armies         int16   (N, map_size)      i16: sim narrows in snapshot getter
        snaps_cities_mask    uint8   (N, map_size)      not bitpacked
        end_ownership        int8    (map_size,)
        end_armies           int32   (map_size,)        i32 at end (not narrowed)
        end_cities_mask      uint8   (map_size,)
        generals             int32   (P,)               tile index per player, -1 if dead
        cities               int32   (C,)               initial + captured/neutralized
        alive                bool    (P,)
        has_kill             bool    (P,)
        alive_count          int32   ()                 0-d scalar
        final_timestep       int32   ()                 0-d scalar
        damage_off_all       int32   (P, P)             offensive damage [from, to]
        captures             int32   (n_captures, 3)    cols: (timestep, captor, captured)
        neutralizes          int32   (n_neutralizes, 2) cols: (timestep, player)
        deaths               int32   (n_deaths, 2)      cols: (timestep, player)

    The asymmetry between i16 snapshot armies and i32 end-state armies mirrors
    the sim's getter shapes. See `sim-core/README.md` for the timestep <->
    snapshot semantics that determine snap_indices.
    """
    snaps_own = state.snapshots_ownership
    snaps_arm = state.snapshots_armies
    snaps_cm = state.snapshots_cities_mask

    return {
        "snap_indices":      np.array(snap_indices, dtype=np.int32),
        "snaps_ownership":   np.stack([snaps_own[t] for t in snap_indices], axis=0),
        "snaps_armies":      np.stack([snaps_arm[t] for t in snap_indices], axis=0),
        "snaps_cities_mask": np.stack([snaps_cm[t] for t in snap_indices], axis=0),
        "end_ownership":     np.asarray(state.ownership, dtype=np.int8),
        "end_armies":        np.asarray(state.armies, dtype=np.int32),
        "end_cities_mask":   np.asarray(state.cities_mask, dtype=np.uint8),
        "generals":          np.asarray(state.generals, dtype=np.int32),
        "cities":            np.asarray(state.cities, dtype=np.int32),
        "alive":             np.asarray(state.alive, dtype=bool),
        "has_kill":          np.asarray(state.has_kill, dtype=bool),
        "alive_count":       np.int32(state.alive_count),
        "final_timestep":    np.int32(state.timestep),
        "damage_off_all":    np.asarray(state.damage_off_all, dtype=np.int32),
        "captures":          _pack_events_3(state.capture_events, "captor", "captured"),
        "neutralizes":       _pack_events_2(state.neutralize_events),
        "deaths":            _pack_events_2(state.death_events),
    }


def _pack_events_3(events, attr_a: str, attr_b: str) -> np.ndarray:
    if not events:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(
        [(e.timestep, getattr(e, attr_a), getattr(e, attr_b)) for e in events],
        dtype=np.int32,
    )


def _pack_events_2(events) -> np.ndarray:
    if not events:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array([(e.timestep, e.player) for e in events], dtype=np.int32)
