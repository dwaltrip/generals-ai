import sqlite3

import numpy as np
import pytest

from replay_parser._collector.config import DB_PATH
from replay_parser._collector.wire import decode as decompress
from replay_parser.decode import (
    Afks,
    GameStatic,
    Moves,
    ReplayData,
    decode_wire,
    decode_wire_array,
)


@pytest.fixture(scope="module")
def sample_wire():
    if not DB_PATH.exists():
        pytest.skip(f"collector DB not found at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT wire_data FROM replays WHERE wire_data IS NOT NULL LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        pytest.skip("no fetched replays in collector DB")
    return row[0]


def test_decode_wire_returns_replay_data(sample_wire):
    rd = decode_wire(sample_wire)
    assert isinstance(rd, ReplayData)
    assert isinstance(rd.static, GameStatic)
    assert isinstance(rd.moves, Moves)
    assert isinstance(rd.afks, Afks)


def test_static_shape(sample_wire):
    rd = decode_wire(sample_wire)
    s = rd.static
    assert s.version >= 1
    assert s.map_width > 0 and s.map_height > 0
    assert len(s.usernames) == len(s.stars) == len(s.initial_generals)
    assert len(s.initial_cities) == len(s.initial_city_armies)
    assert len(s.initial_neutrals) == len(s.initial_neutral_armies)


def test_moves_columnar_dtypes_and_length(sample_wire):
    rd = decode_wire(sample_wire)
    m = rd.moves
    assert m.index.dtype == np.int8
    assert m.source.dtype == np.int16
    assert m.dest.dtype == np.int16
    assert m.is50.dtype == np.uint8
    assert m.timestep.dtype == np.int32
    n = len(m.timestep)
    assert len(m.index) == len(m.source) == len(m.dest) == len(m.is50) == n


def test_afks_columnar_dtypes_and_length(sample_wire):
    rd = decode_wire(sample_wire)
    a = rd.afks
    assert a.index.dtype == np.int8
    assert a.timestep.dtype == np.int32
    assert len(a.index) == len(a.timestep)


def test_columnar_lengths_match_wire(sample_wire):
    """The decoded columnar arrays match the raw wire array lengths."""
    wire = decompress(sample_wire)
    rd = decode_wire_array(wire)
    assert len(rd.moves.timestep) == len(wire[10])
    assert len(rd.afks.timestep) == len(wire[11])


def test_moves_field_values_match_wire(sample_wire):
    """ First row of each columnar field matches wire[10][0]'s positional fields.
        Rename: start→source, end→dest, turn→timestep).
    """
    wire = decompress(sample_wire)
    rd = decode_wire_array(wire)
    if len(wire[10]) == 0:
        pytest.skip("sample replay has no moves")
    wm = wire[10][0]
    assert rd.moves.index[0] == wm[0]
    assert rd.moves.source[0] == wm[1]
    assert rd.moves.dest[0] == wm[2]
    assert rd.moves.is50[0] == wm[3]
    assert rd.moves.timestep[0] == wm[4]


def test_moves_timestep_monotonic(sample_wire):
    """ Wire moves are emitted in non-decreasing timestep order
        Assumed by the cursor-based step loop).
    """
    rd = decode_wire(sample_wire)
    ts = rd.moves.timestep
    assert np.all(np.diff(ts) >= 0), "moves ts array is not monotonically non-decreasing"


def test_empty_moves_and_afks():
    """ An otherwise-valid wire with empty moves/afks decodes to empty arrays
        of the right dtypes.
    """
    fake_wire = [
        18, "fake_id", 10, 10,           # version, id, map dims
        ["a", "b"], [0.0, 0.0],          # usernames, stars
        [], [],                          # cities, cityArmies
        [0, 99],                         # generals
        [],                              # mountains
        [], [],                          # moves, afks (empty)
        None, None,                      # teams, map
        [], [],                          # neutrals, neutralArmies
        [], [], None, [], [],            # swamps, chat, playerColors, lights, settings
        [],                              # modifiers
    ]
    rd = decode_wire_array(fake_wire)
    assert len(rd.moves.timestep) == 0
    assert rd.moves.index.dtype == np.int8
    assert len(rd.afks.timestep) == 0
    assert rd.afks.index.dtype == np.int8
