"""Smoke test: build a sim dict from a live State (using the same plain-numpy
pattern as agent.init_for_game) and confirm it satisfies bc.mask.build_mask's
read surface.

`build_mask` shares the dict shape with `build_obs` but has a much
narrower dependency footprint — no MemoryState, no BFSCache, no
visibility. If `build_mask` runs cleanly on our dict and produces a
sensible legality tensor, the dict shape is correct.
"""

import numpy as np
import pytest

from game_runner import seed_map, sim_adapter
from settings import DB_PATH
import sim_core
from training.bc import mask
from training.bc.constants import H_PADDED, W_PADDED


@pytest.fixture(scope="module")
def state_static_pair():
    if not DB_PATH.exists():
        pytest.skip(f"replay DB not present at {DB_PATH}")
    ids = seed_map.list_two_player_replay_ids(limit=1)
    if not ids:
        pytest.skip("no 2-player replays with wire_data in corpus")
    static = seed_map.load_static_from_db(ids[0])
    state = sim_core.new_state(static)
    return state, static


def _build_sim_dict(state, static):
    return {
        "ownership": [np.asarray(state.snapshots_ownership[0], dtype=np.int8)],
        "armies": [np.asarray(state.snapshots_armies[0], dtype=np.int16)],
        "mountains": np.asarray(static.mountains, dtype=np.int32),
        "initial_cities": np.asarray(static.initial_cities, dtype=np.int32),
        "initial_generals": np.asarray(static.initial_generals, dtype=np.int32),
        "cities": np.asarray(state.cities, dtype=np.int32),
        "cities_present_at": np.asarray(state.cities_present_at, dtype=np.int32),
        "capture_events": sim_adapter.capture_events_to_array(state.capture_events),
    }


def test_build_mask_on_sim_dict(state_static_pair):
    state, static = state_static_pair
    sim = _build_sim_dict(state, static)
    H, W = static.map_height, static.map_width

    m = mask.build_mask(sim, t=0, perspective_slot=0, H=H, W=W)
    assert m.shape == (H_PADDED, W_PADDED, 8)
    assert m.dtype == np.bool_

    # At t=0 the only owned tile for player 0 is its general, which has
    # army=1 — below the army>=2 threshold for legal moves. So the entire
    # mask is False at game start.
    assert not m.any(), "expected no legal moves at t=0 (general has army=1)"
