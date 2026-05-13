import sqlite3

import numpy as np
import pytest

from replay_parser._collector.config import DB_PATH
from replay_parser.decode import decode_wire
from replay_parser.state import build_initial_state


@pytest.fixture(scope="module")
def initial_state():
    if not DB_PATH.exists():
        pytest.skip(f"collector DB not found at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT wire_data FROM replays WHERE wire_data IS NOT NULL AND version = 15 LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        pytest.skip("no fetched v15 replays in collector DB")
    rd = decode_wire(row[0])
    return rd, build_initial_state(rd)


def test_initial_state_invariants(initial_state):
    """Cross-checked invariants on a real replay's t=0 state."""
    rd, st = initial_state
    s = rd.static
    n_cells = s.map_width * s.map_height

    # Grid sizing.
    assert st.ownership.shape == (n_cells,)
    assert st.armies.shape == (n_cells,)
    assert st.cities_mask.shape == (n_cells,)

    # alive_count is consistent with alive[].
    assert st.alive_count == sum(st.alive)
    assert st.num_players == len(s.usernames)

    # All mountains marked with the -2 sentinel.
    assert int((st.ownership == -2).sum()) == len(s.mountains)
    if s.mountains:
        assert np.all(st.ownership[s.mountains] == -2)

    # cities list and cities_mask agree (length and indices).
    assert len(st.cities) == int(st.cities_mask.sum()) == len(s.initial_cities)
    assert all(st.cities_mask[c] for c in st.cities)

    # Cities are neutral at t=0 with their starting armies.
    for c, army in zip(s.initial_cities, s.initial_city_armies, strict=True):
        assert st.ownership[c] == -1
        assert st.armies[c] == army

    # Every alive player has ownership=p and army=1 on their general tile.
    for p, gen in enumerate(st.generals):
        if st.alive[p]:
            assert gen >= 0
            assert st.ownership[gen] == p
            assert st.armies[gen] == 1
