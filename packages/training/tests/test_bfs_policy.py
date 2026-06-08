"""
Targeted tests for `compute_known_passable` — the BFS-passability policy
that combines mountains, structures-in-fog, enemy generals, and the
city army-ratio formula.

Each test hand-builds a tiny 4×4 MemoryState fixture rather than going
through `init_memory` + `step_memory`. The mutation-site assertion lives
here, decoupled from the fog-evolution machinery.

Map fixture:
       c0 c1 c2 c3
  r0:  G0 ·  ·  G1
  r1:  ·  Cn ·  M
  r2:  ·  ·  Ce ·
  r3:  ·  Fs ·  Cs

  M = mountain
  C = city (n = neutral, s = self, e = enemy)
  G0 = self-general
  G1 = enemy-general
"""

from __future__ import annotations

import numpy as np

from training.bc.constants import CITY_TRAVERSABILITY_FACTOR
from training.bc.obs import MemoryState, compute_known_passable
from training.bc.obs_config import OBS_CONFIG_DEFAULTS


H, W = 4, 4
PERSPECTIVE = 0
ENEMY = 1
P = 8

# Cell coords (row, col) — picked to keep the fixture small.
SELF_GENERAL = (0, 0)
ENEMY_GENERAL = (0, 3)
NEUTRAL_CITY = (1, 1)
ENEMY_CITY = (2, 2)
SELF_CITY = (3, 3)
MOUNTAIN = (1, 3)
FOG_STRUCTURE = (3, 1)  # known to be a structure but type not yet resolved
EMPTY_CELL = (2, 0)  # baseline non-structure cell

NEUTRAL_CITY_ARMY = 40
ENEMY_CITY_ARMY = 50
SELF_CITY_ARMY = 10
# Hidden ground-truth army of ENEMY_CITY while it is fogged: it has grown past
# the perspective's last-seen belief (ENEMY_CITY_ARMY). The policy must ignore
# this current value and use last_seen_armies — see the fog branch below.
ENEMY_CITY_FOGGED_ARMY = 999


def _flat(rc: tuple[int, int]) -> int:
    return rc[0] * W + rc[1]


def _make_fixture(total_army: int, *, enemy_city_visible: bool = True):
    """
    Hand-build a (state, vis, own, armies, structures_in_fog_mask) tuple
    for the 4×4 fixture at tick t=0.

    `state.army_count_history[0, PERSPECTIVE]` = total_army drives the formula.
    `enemy_city_visible` toggles whether the formula reads current armies
    (visible path) or `last_seen_armies` (fog path) for ENEMY_CITY.
    """
    is_structure = np.zeros((H, W), dtype=bool)
    for cell in [
        SELF_GENERAL, ENEMY_GENERAL,
        NEUTRAL_CITY, ENEMY_CITY, SELF_CITY,
        MOUNTAIN, FOG_STRUCTURE,
    ]:
        is_structure[cell] = True

    general_locations = np.full(P, -1, dtype=np.int32)
    general_locations[PERSPECTIVE] = _flat(SELF_GENERAL)
    general_locations[ENEMY] = _flat(ENEMY_GENERAL)

    land_row = np.zeros(P, dtype=np.int32)
    army_row = np.zeros(P, dtype=np.int64)
    army_row[PERSPECTIVE] = total_army
    land_count_history = [land_row]
    army_count_history = [army_row]

    known_mountain = np.zeros((H, W), dtype=bool)
    known_city = np.zeros((H, W), dtype=bool)
    known_general = np.zeros((H, W), dtype=bool)
    known_mountain[MOUNTAIN] = True
    known_city[NEUTRAL_CITY] = True
    known_city[ENEMY_CITY] = True
    known_city[SELF_CITY] = True
    known_general[SELF_GENERAL] = True
    known_general[ENEMY_GENERAL] = True
    # FOG_STRUCTURE: in is_structure but not in any known_* → structures_in_fog.

    historically_seen = np.zeros((H, W), dtype=bool)
    for cell in [SELF_GENERAL, SELF_CITY, NEUTRAL_CITY, ENEMY_CITY, ENEMY_GENERAL, MOUNTAIN]:
        historically_seen[cell] = True

    last_seen_owner = np.full((H, W), -2, dtype=np.int8)
    last_seen_armies = np.full((H, W), -1, dtype=np.int16)
    last_seen_owner[SELF_GENERAL] = PERSPECTIVE
    last_seen_armies[SELF_GENERAL] = 1
    last_seen_owner[SELF_CITY] = PERSPECTIVE
    last_seen_armies[SELF_CITY] = SELF_CITY_ARMY
    last_seen_owner[NEUTRAL_CITY] = -1
    last_seen_armies[NEUTRAL_CITY] = NEUTRAL_CITY_ARMY
    last_seen_owner[ENEMY_CITY] = ENEMY
    last_seen_armies[ENEMY_CITY] = ENEMY_CITY_ARMY
    last_seen_owner[ENEMY_GENERAL] = ENEMY
    last_seen_armies[ENEMY_GENERAL] = 1

    state = MemoryState(
        obs_cfg=OBS_CONFIG_DEFAULTS,
        perspective_slot=PERSPECTIVE,
        opp_slots=[s for s in range(P) if s != PERSPECTIVE],
        is_structure=is_structure,
        general_locations=general_locations,
        land_count_history=land_count_history,
        army_count_history=army_count_history,
        historically_seen=historically_seen,
        known_mountain=known_mountain,
        known_city=known_city,
        known_general=known_general,
        last_seen_owner=last_seen_owner,
        last_seen_armies=last_seen_armies,
        turns_since_seen=np.full((H, W), -1, dtype=np.int32),
        opp_has_seen=np.zeros((P, H, W), dtype=bool),
        opp_contacted=np.zeros(P, dtype=bool),
        opp_captured_by=np.full(P, -1, dtype=np.int8),
    )

    vis = np.zeros((H, W), dtype=bool)
    vis[SELF_GENERAL] = True
    vis[SELF_CITY] = True
    if enemy_city_visible:
        vis[ENEMY_CITY] = True

    own = np.full((H, W), -1, dtype=np.int8)
    armies = np.zeros((H, W), dtype=np.int16)
    own[SELF_GENERAL] = PERSPECTIVE
    own[ENEMY_GENERAL] = ENEMY
    own[NEUTRAL_CITY] = -1
    own[ENEMY_CITY] = ENEMY
    own[SELF_CITY] = PERSPECTIVE
    armies[SELF_GENERAL] = 1
    armies[SELF_CITY] = 10
    armies[NEUTRAL_CITY] = NEUTRAL_CITY_ARMY
    # When visible, the current army equals last-seen (step_memory pins them on
    # every sighting). When fogged, the current array holds the hidden ground
    # truth (999), which diverges from the stale last-seen (50) — so the fog
    # test can tell whether the policy reads the right source.
    armies[ENEMY_CITY] = (
        ENEMY_CITY_ARMY if enemy_city_visible else ENEMY_CITY_FOGGED_ARMY
    )
    armies[ENEMY_GENERAL] = 1

    structures_in_fog_mask = (
        is_structure & ~known_mountain & ~known_city & ~known_general
    )
    return state, vis, own, armies, structures_in_fog_mask


def _passable(args, cell: tuple[int, int]) -> bool:
    state, vis, own, armies, sfm = args
    flat = compute_known_passable(state, 0, PERSPECTIVE, vis, own, armies, sfm, H, W)
    return bool(flat[_flat(cell)])


def test_mountains_impassable():
    assert not _passable(_make_fixture(total_army=10_000), MOUNTAIN)


def test_structures_in_fog_impassable():
    assert not _passable(_make_fixture(total_army=10_000), FOG_STRUCTURE)


def test_own_general_always_passable():
    # Own general is in `known_general`; must be excluded from enemy-general mask.
    assert _passable(_make_fixture(total_army=1), SELF_GENERAL)


def test_enemy_general_always_impassable():
    assert not _passable(_make_fixture(total_army=10_000), ENEMY_GENERAL)


def test_own_city_passable_regardless_of_total_army():
    # Own city is in `known_city`; must skip the formula via the owner check.
    assert _passable(_make_fixture(total_army=1), SELF_CITY)


def test_empty_cell_passable():
    # Sanity: a non-structure cell stays passable.
    assert _passable(_make_fixture(total_army=1), EMPTY_CELL)


def test_neutral_city_at_threshold_impassable():
    # threshold = 40 * 4 = 160. Strict-greater means total_army=160 → impassable.
    threshold = NEUTRAL_CITY_ARMY * CITY_TRAVERSABILITY_FACTOR
    assert not _passable(_make_fixture(total_army=threshold), NEUTRAL_CITY)


def test_neutral_city_above_threshold_passable():
    threshold = NEUTRAL_CITY_ARMY * CITY_TRAVERSABILITY_FACTOR
    assert _passable(_make_fixture(total_army=threshold + 1), NEUTRAL_CITY)


def test_enemy_city_visible_uses_current_armies():
    threshold = ENEMY_CITY_ARMY * CITY_TRAVERSABILITY_FACTOR  # 200
    assert not _passable(_make_fixture(total_army=threshold, enemy_city_visible=True), ENEMY_CITY)
    assert _passable(_make_fixture(total_army=threshold + 1, enemy_city_visible=True), ENEMY_CITY)


def test_enemy_city_fogged_uses_last_seen_armies():
    # Fogged: current army is the hidden 999, last-seen is 50. Threshold is built
    # from the last-seen value, so these pass only if the policy ignores 999. A
    # bug reading the current army would push the threshold to 999*4 and flip the
    # +1 case to impassable.
    threshold = ENEMY_CITY_ARMY * CITY_TRAVERSABILITY_FACTOR
    assert not _passable(_make_fixture(total_army=threshold, enemy_city_visible=False), ENEMY_CITY)
    assert _passable(_make_fixture(total_army=threshold + 1, enemy_city_visible=False), ENEMY_CITY)
