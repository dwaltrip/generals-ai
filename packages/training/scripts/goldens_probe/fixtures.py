"""Produce-fixture core: one corpus replay → the sim/meta npz pair, probe-grade.

Mirrors the corpus driver's per-game flow (`replay_parser.driver._process_replay_inner`)
minus the curated-player and rolling-stats gating: a probe fixture records *every*
slot as a perspective, with real placement from the DB listing join.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

import numpy as np

from replay_parser._collector.wire import decode as decompress_wire
from replay_parser._shared import is_vanilla_ffa
from replay_parser.decode import decode_wire_array
from replay_parser.metadata import build_metadata
from replay_parser.output import write_metadata, write_sim_output
import sim_core


class FixtureSkip(Exception):
    """This replay can't become a fixture (wire filter, missing data, name mismatch)."""


@dataclass(frozen=True)
class Candidate:
    replay_id: str
    turns: int
    player_count: int


def candidate_replays(conn: sqlite3.Connection) -> list[Candidate]:
    """Replays passing the driver's SQL-level filter, in deterministic order."""
    rows = conn.execute(
        """
        SELECT id, turns, player_count
        FROM replays
        WHERE ladder_id = 'ffa'
          AND version >= 15
          AND player_count BETWEEN 4 AND 8
          AND wire_data IS NOT NULL
        ORDER BY started ASC, id ASC
        """
    ).fetchall()
    return [Candidate(rid, int(t or 0), int(pc)) for rid, t, pc in rows]


def produce_fixture(
    conn: sqlite3.Connection, replay_id: str, out_dir: Path
) -> tuple[Path, Path]:
    """Parse one replay and write `<id>.npz` + `<id>.meta.npz` under `out_dir`."""
    row = conn.execute(
        "SELECT wire_data FROM replays WHERE id = ?", (replay_id,)
    ).fetchone()
    if row is None or row[0] is None:
        raise FixtureSkip("no wire_data")
    wire = decompress_wire(row[0])
    if not is_vanilla_ffa(wire):
        raise FixtureSkip("wire filter (not vanilla FFA)")
    replay = decode_wire_array(wire)
    state = sim_core.simulate(replay)

    placement_rows = conn.execute(
        "SELECT rp.position, p.name FROM replay_players rp "
        "JOIN players p ON rp.player_id = p.id WHERE rp.replay_id = ?",
        (replay_id,),
    ).fetchall()
    # Listing position is 0-indexed; placement is 1-indexed (as in the driver).
    placement_by_name = {name: pos + 1 for pos, name in placement_rows}

    usernames = replay.static.map.usernames
    missing = [n for n in usernames if n not in placement_by_name]
    if missing:
        raise FixtureSkip(f"listing/wire name mismatch: {missing[:3]!r}")

    meta = build_metadata(
        state,
        replay,
        perspective_player_ids=list(range(len(usernames))),
        placement=[placement_by_name[n] for n in usernames],
        sim_core_version="goldens-probe",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_path = out_dir / f"{replay_id}.npz"
    meta_path = out_dir / f"{replay_id}.meta.npz"
    write_sim_output(state, replay, sim_path)
    write_metadata(meta, meta_path)
    return sim_path, meta_path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def event_summary(sim: dict[str, np.ndarray]) -> dict[str, int]:
    return {
        "T": int(sim["ownership"].shape[0]),
        "H": int(sim["map_height"]),
        "W": int(sim["map_width"]),
        "captures": int(sim["capture_events"].shape[0]),
        "deaths": int(sim["death_events"].shape[0]),
        "neutralizes": int(sim["neutralize_events"].shape[0]),
    }
