"""Phase 2 smoke test.

Verifies that:
  - The uv workspace resolves the replay-parser → replay-collector edge.
  - The `_collector/wire.py` bridge re-exports `decode` from `replay_collector.wire`.
  - The parser can read a `wire_data` BLOB out of the collector's sqlite DB
    and decode it end-to-end.
"""

import sqlite3

import pytest

from replay_parser._collector.config import DB_PATH
from replay_parser._collector.wire import decode


@pytest.mark.skipif(not DB_PATH.exists(), reason=f"collector DB not found at {DB_PATH}")
def test_decode_one_wire_blob_from_collector_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            """SELECT wire_data FROM replays
            WHERE wire_data IS NOT NULL and version = 15 LIMIT 1"""
        ).fetchone()
    finally:
        conn.close()

    assert row is not None, "no fetched replays in the collector DB"
    wire = decode(row[0])
    assert isinstance(wire, list)
    assert wire[0] >= 15, f"unexpected wire version: {wire[0]}"
