"""Smoke test: build the sim dict from a live State and confirm it satisfies
bc.mask.build_mask's read surface.

`build_mask` shares the dict shape with `build_obs` but has a much
narrower dependency footprint — no MemoryState, no BFSCache, no
visibility. If `build_mask` runs cleanly on our adapter output and
produces a sensible legality tensor, the dict shape is correct.
"""

import numpy as np
import pytest

import sim_core

from bc import mask
from bc.constants import H_PADDED, W_PADDED
from replay_collector.config import DB_PATH
from self_play import seed_map, sim_adapter


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


def test_dict_shape(state_static_pair):
    state, static = state_static_pair
    sim = sim_adapter.sim_dict_from_state(state, static, max_turns_hint=1000)

    expected_keys = {
        "ownership", "armies", "mountains", "initial_cities",
        "initial_generals", "cities", "cities_present_at", "capture_events",
    }
    assert set(sim.keys()) == expected_keys

    # Time-indexed fields satisfy [t] + .shape[0] access pattern.
    map_size = static.map_width * static.map_height
    own_t = sim["ownership"][0]
    assert own_t.shape == (map_size,)
    assert own_t.dtype == np.int8
    assert sim["ownership"].shape == (1000,)

    arm_t = sim["armies"][0]
    assert arm_t.shape == (map_size,)
    assert arm_t.dtype == np.int16
    assert sim["armies"].shape == (1000,)


def test_build_mask_runs_on_adapter(state_static_pair):
    state, static = state_static_pair
    sim = sim_adapter.sim_dict_from_state(state, static)
    H, W = static.map_height, static.map_width

    m = mask.build_mask(sim, t=0, perspective_slot=0, H=H, W=W)
    assert m.shape == (H_PADDED, W_PADDED, 8)
    assert m.dtype == np.bool_

    # At t=0 the only owned tile for player 0 is its general, which has
    # army=1 — below the army>=2 threshold for legal moves. So the entire
    # mask is False at game start. This isn't a useful state to play
    # from, but it confirms the legality logic is wired correctly to our
    # adapter's ownership/armies/mountains fields.
    assert not m.any(), "expected no legal moves at t=0 (general has army=1)"
