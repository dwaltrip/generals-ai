"""Smoke test: corpus → static → sim_core.new_state pipeline.

End-to-end check that the seed_map path produces a buildable State for
sim_core. Skips if the local replay DB isn't populated.
"""

import pytest

import sim_core

from replay_collector.config import DB_PATH
from self_play import seed_map


@pytest.fixture(scope="module")
def two_player_replay_id() -> str:
    if not DB_PATH.exists():
        pytest.skip(f"replay DB not present at {DB_PATH}")
    ids = seed_map.list_two_player_replay_ids(limit=1)
    if not ids:
        pytest.skip("no 2-player replays with wire_data in corpus")
    return ids[0]


def test_seed_map_to_new_state(two_player_replay_id):
    static = seed_map.load_static_from_db(two_player_replay_id)
    state = sim_core.new_state(static)

    assert state.num_players == 2
    assert state.alive_count == 2
    assert state.alive == [True, True]
    assert state.timestep == 0

    # snapshot indexing: new_state writes the initial snapshot, so we should
    # have exactly one frame of history at t=0.
    assert state.snapshots_len == 1

    # ownership grid sized to the static's map dims, generals placed.
    map_size = static.map_width * static.map_height
    assert state.ownership.shape == (map_size,)
    assert state.armies.shape == (map_size,)
    # Each general tile is owned by its player and has army=1.
    for p, gen_idx in enumerate(state.generals):
        assert state.ownership[gen_idx] == p
        assert state.armies[gen_idx] == 1
