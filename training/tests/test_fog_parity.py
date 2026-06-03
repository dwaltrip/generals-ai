"""
Hand-fixture fog parity test for `MemoryState` evolution.

Drives a small synthetic game (5x6 board, 2 players, 8 snapshots) through
`init_memory` + `step_memory` and asserts the running state matches
hand-computed expectations at key checkpoint ticks.

The point of this fixture is to be the *correctness oracle* for the fog
memory logic. The dataloader smoke test catches shape/NaN bugs; this test
catches silent semantic bugs (wrong sentinel values, monotonic accumulator
regressions, mishandled capture events, etc.) that would silently corrupt
the obs tensor going into training.

Board geometry (see also: training/tests/fixtures/small-1.txt):

  row 0:  ·  M  ·  ·  ·  M
  row 1:  ·  ·  ·  ·  G1 ·
  row 2:  C  M  ·  M  ·  M
  row 3:  ·  ·  ·  ·  C  ·
  row 4:  G0 ·  M  ·  M  ·

  M = mountain (7 cells)
  C = initial neutral city (2 cells)
  G0 = perspective general at (4, 0), slot 0
  G1 = opponent general at (1, 4), slot 1
  · = neutral plain

Trajectory: perspective expands one cell per tick toward the opponent.
Structures resolve as they enter the Moore-neighborhood. At action t=6
the perspective captures the opponent; snapshot[7] reflects the capture.

Known limitations of this fixture (worth a follow-up):
  - No "visible → fogged" transition; `turns_since_seen` increment logic
    is exercised only for the "stays visible (TSS=0)" and "never seen
    (TSS=-1)" cases. Visible-then-fogged needs a separate small fixture.
  - Slot canonicalization: this fixture uses perspective=slot-0 only.
    Verifying canonical-channel mapping for non-zero perspective slots
    (essential #3) needs a separate fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from bc.obs import init_memory, step_memory
from bc.obs_config import OBS_CONFIG_DEFAULTS
from bc.visibility import compute_visibility


H, W = 5, 6
HW = H * W

PERSPECTIVE = 0
OPP = 1
P_FIXTURE = 2  # Two-player fixture; init_memory + step_memory both take P as kwarg.


def _cell(r: int, c: int) -> int:
    """(row, col) → flat unpadded cell index."""
    return r * W + c


# ---------------------------------------------------------------------------
# Static fixture data (matches small-1.txt)
# ---------------------------------------------------------------------------

_MOUNTAINS = [
    _cell(0, 1), _cell(0, 5),
    _cell(2, 1), _cell(2, 3), _cell(2, 5),
    _cell(4, 2), _cell(4, 4),
]
_INITIAL_CITIES = [_cell(2, 0), _cell(3, 4)]
_INITIAL_GENERALS = [_cell(4, 0), _cell(1, 4)]  # index = raw slot id

# Per-tick ownership trajectory. Hand-crafted (we do NOT route this through
# sim_core); each tick the perspective accumulates one new cell.
_PERSPECTIVE_OWNED = [
    [_cell(4, 0)],
    [_cell(4, 0), _cell(4, 1)],
    [_cell(4, 0), _cell(4, 1), _cell(3, 1)],
    [_cell(4, 0), _cell(4, 1), _cell(3, 1), _cell(3, 2)],
    [_cell(4, 0), _cell(4, 1), _cell(3, 1), _cell(3, 2), _cell(3, 3)],
    [_cell(4, 0), _cell(4, 1), _cell(3, 1), _cell(3, 2), _cell(3, 3), _cell(2, 4)],
    # snapshot[6] is pre-action: opp still owns (1, 4). The capture event
    # fires at action t=6, so opp_captured_by updates during step_memory(6)
    # but ownership[6, (1,4)] is still slot 1 until snapshot[7].
    [_cell(4, 0), _cell(4, 1), _cell(3, 1), _cell(3, 2), _cell(3, 3), _cell(2, 4)],
    # snapshot[7]: post-capture. Perspective now owns the former opp general.
    [_cell(4, 0), _cell(4, 1), _cell(3, 1), _cell(3, 2), _cell(3, 3), _cell(2, 4), _cell(1, 4)],
]
# Opp owns just their general until captured.
_OPP_OWNED = [
    [_cell(1, 4)],
    [_cell(1, 4)],
    [_cell(1, 4)],
    [_cell(1, 4)],
    [_cell(1, 4)],
    [_cell(1, 4)],
    [_cell(1, 4)],
    [],
]

T = len(_PERSPECTIVE_OWNED)  # 8


# ---------------------------------------------------------------------------
# Sim-dict construction
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim() -> dict[str, np.ndarray]:
    """
    Construct the sim dict programmatically. Matches the parser-output schema
    closely enough that init_memory / step_memory see what they'd see in
    production.

    NOTE: we don't include every parser-output field — only the ones consumed
    by the code under test. If init_memory / step_memory grow to consume more
    fields, this fixture needs to grow with them.
    """
    ownership = np.full((T, HW), -1, dtype=np.int8)  # default: neutral
    armies = np.zeros((T, HW), dtype=np.int16)

    for t in range(T):
        for m in _MOUNTAINS:
            ownership[t, m] = -2
        for c in _PERSPECTIVE_OWNED[t]:
            ownership[t, c] = PERSPECTIVE
            armies[t, c] = 10  # arbitrary; the test doesn't assert on specific army values
        for c in _OPP_OWNED[t]:
            ownership[t, c] = OPP
            armies[t, c] = 1
        # Neutral cities have non-zero army (~40 in real games). Doesn't matter
        # for fog state — the test only checks last_seen_armies sentinels.
        for c in _INITIAL_CITIES:
            if ownership[t, c] == -1:
                armies[t, c] = 40

    # Cities: initial + the new city from capture at action t=6 → snapshot[7].
    # cities_present_at uses the SNAPSHOT INDEX (0 for initial, t+1 for the
    # snapshot where the new city first appears).
    cities = np.array(
        [_cell(2, 0), _cell(3, 4), _cell(1, 4)], dtype=np.int32
    )
    cities_present_at = np.array([0, 0, 7], dtype=np.int32)

    # Capture event: timestamp = action step. (6, captor=0, captured=1).
    capture_events = np.array([[6, PERSPECTIVE, OPP]], dtype=np.int32)

    return {
        "ownership": ownership,
        "armies": armies,
        "mountains": np.array(_MOUNTAINS, dtype=np.int32),
        "initial_cities": np.array(_INITIAL_CITIES, dtype=np.int32),
        "initial_generals": np.array(_INITIAL_GENERALS, dtype=np.int32),
        "cities": cities,
        "cities_present_at": cities_present_at,
        "capture_events": capture_events,
        "map_width": np.int32(W),
        "map_height": np.int32(H),
    }


def _walk_to(sim: dict[str, np.ndarray], target_t: int):
    """Run init_memory + step_memory for t=0..target_t inclusive."""
    state = init_memory(sim, PERSPECTIVE, H, W, OBS_CONFIG_DEFAULTS, P=P_FIXTURE)
    for t in range(target_t + 1):
        vis = compute_visibility(sim["ownership"][t], PERSPECTIVE, H, W)
        step_memory(state, sim, t, vis, PERSPECTIVE, H, W, P=P_FIXTURE)
    return state


def _mask_with(coords: list[tuple[int, int]]) -> np.ndarray:
    """Build a bool [H, W] mask with True at the given (r, c) coords."""
    m = np.zeros((H, W), dtype=bool)
    for r, c in coords:
        m[r, c] = True
    return m


# ===========================================================================
# Checkpoint: t=0 — init state, only perspective's general visible
# ===========================================================================


def test_t0_init_state(sim):
    state = _walk_to(sim, 0)

    # Vision at t=0 = Moore-neighborhood of (4, 0), clipped to bounds.
    np.testing.assert_array_equal(
        state.historically_seen,
        _mask_with([(3, 0), (3, 1), (4, 0), (4, 1)]),
    )

    # No structures resolved yet — perspective hasn't seen any structure tiles.
    np.testing.assert_array_equal(state.known_mountain, np.zeros((H, W), dtype=bool))
    np.testing.assert_array_equal(state.known_city, np.zeros((H, W), dtype=bool))
    # Only own general known (set in init_memory at t=0).
    np.testing.assert_array_equal(state.known_general, _mask_with([(4, 0)]))

    # General-location source pins.
    assert state.general_locations[PERSPECTIVE] == _cell(4, 0)
    assert state.general_locations[OPP] == -1  # opp unknown

    # Contact / capture state.
    assert not state.opp_contacted[OPP]
    assert state.opp_captured_by[OPP] == -1  # alive

    # Sentinel checks for a cell that's never been seen.
    assert state.last_seen_owner[0, 0] == -2
    assert state.last_seen_armies[0, 0] == -1
    assert state.turns_since_seen[0, 0] == -1


# ===========================================================================
# Checkpoint: t=2 — first structure resolutions (mountain + city)
#   - Mountain at (4, 2) resolved at t=1 (visible from (4, 1)).
#   - Mountain at (2, 1) resolved at t=2.
#   - City at (2, 0) resolved at t=2 → first BFS graph growth.
# ===========================================================================


def test_t2_first_structure_resolutions(sim):
    state = _walk_to(sim, 2)

    # Cumulative vision through t=2 (union of Moore-nbhds at t=0,1,2).
    np.testing.assert_array_equal(
        state.historically_seen,
        _mask_with([
            (2, 0), (2, 1), (2, 2),
            (3, 0), (3, 1), (3, 2),
            (4, 0), (4, 1), (4, 2),
        ]),
    )

    # Mountains resolved: (4, 2) at t=1; (2, 1) at t=2.
    np.testing.assert_array_equal(
        state.known_mountain, _mask_with([(4, 2), (2, 1)]),
    )
    # City at (2, 0) resolved at t=2.
    np.testing.assert_array_equal(state.known_city, _mask_with([(2, 0)]))
    # No new generals.
    np.testing.assert_array_equal(state.known_general, _mask_with([(4, 0)]))

    # Last-seen owner at a few cells:
    #   (2, 0): neutral city visible at t=2 → -1 (neutral).
    #   (0, 0): never seen → -2 sentinel.
    #   (4, 0): perspective's general, always visible.
    #   (4, 1): perspective owns it from t=1 onward.
    assert state.last_seen_owner[2, 0] == -1
    assert state.last_seen_owner[0, 0] == -2
    assert state.last_seen_owner[4, 0] == PERSPECTIVE
    assert state.last_seen_owner[4, 1] == PERSPECTIVE

    # No opp contact yet — (1, 4) far from current vision.
    assert not state.opp_contacted[OPP]
    assert state.opp_captured_by[OPP] == -1
    # opp_has_seen[OPP] is empty until perspective directly sights an opp tile.
    np.testing.assert_array_equal(
        state.opp_has_seen[OPP], np.zeros((H, W), dtype=bool),
    )


# ===========================================================================
# Checkpoint: t=5 — opp general discovered
#   - Vision reaches (2, 4); Moore-nbhd of (2, 4) includes (1, 4).
#   - (1, 4) is the opp general → known_general fires, general_locations
#     pin updates, opp_contacted flips.
# ===========================================================================


def test_t5_opp_general_discovered(sim):
    state = _walk_to(sim, 5)

    # Opp general now in known_general.
    np.testing.assert_array_equal(
        state.known_general, _mask_with([(4, 0), (1, 4)]),
    )
    # Source pin updated.
    assert state.general_locations[OPP] == _cell(1, 4)
    # Contact flipped True.
    assert state.opp_contacted[OPP]
    # Last-seen owner at (1, 4): opp (just saw it).
    assert state.last_seen_owner[1, 4] == OPP

    # Capture state unchanged — capture event is at action t=6.
    assert state.opp_captured_by[OPP] == -1

    # Mountains accumulated: (4,2)@t=1, (2,1)@t=2, (2,3)@t=3, (4,4)@t=4, (2,5)@t=5.
    np.testing.assert_array_equal(
        state.known_mountain,
        _mask_with([(4, 2), (2, 1), (2, 3), (4, 4), (2, 5)]),
    )
    # Cities: (2, 0)@t=2 + (3, 4)@t=4.
    np.testing.assert_array_equal(
        state.known_city, _mask_with([(2, 0), (3, 4)]),
    )

    # opp_has_seen[OPP] activates at t=5: perspective takes (2,4), whose
    # Moore-nbhd includes (1,4) — opp's general. The dilation around (1,4)
    # marks the 3x3 block (0,3)..(2,5) (all in-bounds on 5x6 board).
    np.testing.assert_array_equal(
        state.opp_has_seen[OPP],
        _mask_with([
            (0, 3), (0, 4), (0, 5),
            (1, 3), (1, 4), (1, 5),
            (2, 3), (2, 4), (2, 5),
        ]),
    )


# ===========================================================================
# Checkpoint: t=6 — capture event processed
#   - opp_captured_by[OPP] flips immediately (step_memory processes events
#     with timestamp == t).
#   - Snapshot[6] is pre-action: ownership at (1, 4) is still OPP.
#   - cities_present_at[(1, 4)] = 7 > 6 → known_city NOT yet updated.
# ===========================================================================


def test_t6_capture_event_processed(sim):
    state = _walk_to(sim, 6)

    assert state.opp_captured_by[OPP] == PERSPECTIVE  # captured by self (slot 0)

    # Pre-action snapshot still shows opp owning their general.
    assert state.last_seen_owner[1, 4] == OPP

    # known_city at (1, 4) not yet — that's a t=7 event.
    assert not state.known_city[1, 4]
    # known_general at (1, 4) still True — opp general is still visible.
    assert state.known_general[1, 4]


# ===========================================================================
# Checkpoint: t=7 — post-capture state
#   - ownership[7, (1, 4)] = PERSPECTIVE (cell flipped to captor as a city).
#   - cities_present_at = 7 → cities_at_t includes (1, 4) at this step.
#   - new_city fires at (1, 4) → known_city updated.
#   - is_general_now excludes (1, 4) (it's now in cities_at_t) → new_general
#     does NOT fire. known_general STAYS True at (1, 4) — this is the
#     documented stale-flag behavior (TODO in step_memory).
# ===========================================================================


def test_t7_post_capture(sim):
    state = _walk_to(sim, 7)

    # Captured general cell now resolves as a city.
    assert state.known_city[1, 4]

    # Documented stale-flag behavior: known_general at (1, 4) stays True even
    # though the cell is now a city. Flip this assertion when the stale-flag
    # TODO in step_memory is fixed.
    assert state.known_general[1, 4]

    # Ownership at (1, 4) is now perspective (the captor owns the new city).
    assert state.last_seen_owner[1, 4] == PERSPECTIVE

    # Mountain at (0, 5) becomes visible at t=7 (now in Moore-nbhd of (1, 4)).
    assert state.known_mountain[0, 5]

    # (0, 0) is still never seen by t=7.
    assert state.last_seen_owner[0, 0] == -2

    # opp_has_seen[OPP] hasn't grown since t=5: t=6's vision over (1,4) is
    # the same Moore-nbhd (no new cells added); at t=7 opp owns nothing
    # visible, so no further dilation fires.
    np.testing.assert_array_equal(
        state.opp_has_seen[OPP],
        _mask_with([
            (0, 3), (0, 4), (0, 5),
            (1, 3), (1, 4), (1, 5),
            (2, 3), (2, 4), (2, 5),
        ]),
    )
    assert state.turns_since_seen[0, 0] == -1
