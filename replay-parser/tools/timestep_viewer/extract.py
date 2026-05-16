"""Snapshot extractor for the timestep viewer.

Given a replay ID, queries the collector DB for its wire_data blob, runs
parse_replay to get a finished sim State, and converts it into a JSON-ready
dict for embedding in the viewer HTML.

Data shape rationale:
  - Per-timestep snapshots are base64-encoded typed-array bytes for
    `ownership` (int8) and `armies` (int16). cities_mask is omitted —
    cities are monotonic (initial_cities + a small stream of city-births
    from captures and neutralizes), so the per-timestep mask is wasteful.
  - `city_births` covers both capture and neutralize events; both convert
    a general tile into a city tile. The user-facing event log uses a
    separate `ui_events` list that suppresses neutralize entries (the
    official viewer doesn't surface them).
  - `general_death_t[p]` = the sim-timestep at which the action capturing
    or neutralizing player p occurred, or null if the player still has
    their general at game-end. The general tile is still drawn as a
    general in SNAPS[general_death_t[p]] (that's the pre-resolution
    snapshot); it flips to a city starting at SNAPS[general_death_t[p]+1].
    JS rule: a tile is a general at displayed step t iff some p has
    initial_generals[p] == tile AND (general_death_t[p] is null OR
    t <= general_death_t[p]).
"""
from __future__ import annotations

from base64 import b64encode
import sqlite3

import sim_core

from replay_parser._collector.config import DB_PATH
from replay_parser.parser import parse_replay


def extract(replay_id: str, conn: sqlite3.Connection | None = None) -> dict:
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT wire_data FROM replays WHERE id = ?", (replay_id,)
        ).fetchone()
    finally:
        if own_conn:
            conn.close()
    if row is None or row[0] is None:
        raise ValueError(f"no wire_data for replay_id={replay_id!r}")
    state, replay = parse_replay(row[0])
    return _build_payload(replay_id, state, replay)


def _build_payload(replay_id: str, state: sim_core.State, replay) -> dict:
    s = replay.static
    initial_generals = list(s.initial_generals)

    # City-birth stream: every capture and neutralize event converts the
    # affected player's general tile to a city. Order by timestep, then by
    # event ordering within the timestep (captures before neutralizes —
    # the sim never produces both for the same player at the same t, but
    # a tie-break keeps the JS-side fold deterministic).
    city_births: list[dict] = []
    for ce in state.capture_events:
        tile = initial_generals[ce.captured]
        if tile >= 0:
            city_births.append({"t": ce.timestep, "tile": tile})
    for ne in state.neutralize_events:
        tile = initial_generals[ne.player]
        if tile >= 0:
            city_births.append({"t": ne.timestep, "tile": tile})
    city_births.sort(key=lambda c: c["t"])

    # general_death_t[p]: the timestep at which player p's general becomes
    # a city (via capture or neutralize), or None if p ends the game still
    # holding their general.
    general_death_t: list[int | None] = [None] * state.num_players
    for ce in state.capture_events:
        general_death_t[ce.captured] = ce.timestep
    for ne in state.neutralize_events:
        # Neutralize can fire after capture for the same player only in
        # pathological cases (shouldn't, post the eb239be fix). Keep the
        # earlier timestep if both apply.
        prev = general_death_t[ne.player]
        general_death_t[ne.player] = (
            ne.timestep if prev is None else min(prev, ne.timestep)
        )

    # User-facing event log: captures + surrenders + winner. Surrenders
    # are DeathEvents that don't have a matching CaptureEvent at the same
    # timestep (the captured player's death is implicit in the capture
    # entry).
    captured_keys = {(ce.captured, ce.timestep) for ce in state.capture_events}
    ui_events: list[dict] = []
    for ce in state.capture_events:
        ui_events.append({
            "t": ce.timestep,
            "kind": "capture",
            "captor": ce.captor,
            "captured": ce.captured,
        })
    for de in state.death_events:
        if (de.player, de.timestep) in captured_keys:
            continue
        ui_events.append({
            "t": de.timestep,
            "kind": "surrender",
            "player": de.player,
        })
    alive_at_end = [p for p in range(state.num_players) if state.alive[p]]
    if len(alive_at_end) == 1:
        # state.timestep at loop exit is one past the last step that ran;
        # subtract 1 so the win appears at the same displayed step as the
        # deciding action under the JS-side `e.t < stepT` rule.
        ui_events.append({
            "t": state.timestep - 1,
            "kind": "win",
            "player": alive_at_end[0],
        })
    ui_events.sort(key=lambda e: (e["t"], e["kind"]))

    snapshots: list[dict] = []
    for t in range(state.snapshots_len):
        ownership = state.snapshots_ownership[t]   # int8[map_size]
        armies = state.snapshots_armies[t]         # int16[map_size]
        snapshots.append({
            "o": b64encode(ownership.tobytes()).decode("ascii"),
            "a": b64encode(armies.tobytes()).decode("ascii"),
        })

    static = {
        "replay_id": replay_id,
        "map_w": s.map_width,
        "map_h": s.map_height,
        "usernames": list(s.usernames),
        "mountains": list(s.mountains),
        "initial_cities": list(s.initial_cities),
        "initial_city_armies": list(s.initial_city_armies),
        "initial_generals": initial_generals,
        "num_timesteps": state.snapshots_len,
        "general_death_t": general_death_t,
    }
    return {
        "static": static,
        "snapshots": snapshots,
        "ui_events": ui_events,
        "city_births": city_births,
    }
