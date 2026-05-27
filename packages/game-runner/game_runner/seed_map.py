"""Seed a self-play game from a real replay's static-map record.

The sim_core sim takes a static map (mountains, initial cities + armies,
initial generals, neutrals) as input — generals.io's server produces
these at game start. We don't have a server-side generator, so for the
prototype we pluck a static from an existing corpus replay and feed it
to `sim_core.new_state(...)`.

`load_static_from_db(replay_id)` is the one we'll actually call from the
self-play loop. `list_two_player_replay_ids(...)` is a convenience for
picking a candidate without hand-writing SQL.

Uses `decode_wire` directly rather than `parse_replay` — the latter
re-runs `sim_core.simulate()` which we don't need here (we only want
the parsed static).
"""

import sqlite3
from typing import Any

from replay_collector.config import DB_PATH
from replay_parser.decode import decode_wire


def load_static_from_db(replay_id: str) -> Any:
    """Fetch a replay's wire blob and return its decoded `.static`.

    The returned object is a `ReplayStatic` record with the fields
    `sim_core.new_state` consumes: map_width, map_height, usernames,
    mountains, initial_cities, initial_city_armies, initial_generals,
    initial_neutrals, initial_neutral_armies.
    """
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT wire_data FROM replays WHERE id = ?",
            (replay_id,),
        ).fetchone()
    if row is None:
        raise LookupError(f"no replay with id={replay_id!r}")
    wire = row[0]
    if wire is None:
        raise LookupError(f"replay {replay_id!r} has no wire_data (metadata-only row)")
    return decode_wire(wire).static


def list_two_player_replay_ids(limit: int = 20) -> list[str]:
    """Candidate 2-player replay IDs, longest-game first.

    Longer games are generally more interesting starting positions to
    seed self-play from (more developed maps, real generals.io 1v1
    dynamics rather than instant-surrender lobbies).
    """
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT id FROM replays
            WHERE player_count = 2 AND wire_data IS NOT NULL
            ORDER BY turns DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [r[0] for r in rows]
