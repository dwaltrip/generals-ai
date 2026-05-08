import atexit
import datetime as dt
import signal
import sqlite3
import sys
import time
from pathlib import Path

from replay_collector import wire


def format_started_date(started: int | None) -> str:
    """`started` is stored as-is from the listing API: epoch ms in practice,
    but very old replays may be epoch-seconds. The 1e12 boundary distinguishes
    them (≈ 2001-09 in ms, ≈ 33658 AD in seconds)."""
    if started is None:
        return "?"
    seconds = started / 1000 if started > 1e12 else started
    return dt.datetime.fromtimestamp(seconds).strftime("%Y-%m-%d")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "generals.sqlite"

_NOW_MS = "(CAST(unixepoch('subsec') * 1000 AS INTEGER))"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS players (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    created_at INTEGER NOT NULL DEFAULT {_NOW_MS},
    updated_at INTEGER NOT NULL DEFAULT {_NOW_MS}
);

CREATE TRIGGER IF NOT EXISTS trg_players_updated_at
AFTER UPDATE ON players
BEGIN
    UPDATE players SET updated_at = {_NOW_MS} WHERE id = NEW.id;
END;

CREATE TABLE IF NOT EXISTS replays (
    id            TEXT PRIMARY KEY,
    -- from listing API (always populated)
    type          TEXT NOT NULL,
    ladder_id     TEXT,
    started       INTEGER NOT NULL,
    turns         INTEGER NOT NULL,
    player_count  INTEGER NOT NULL,
    -- from .gior fetch (null until fetched)
    version       INTEGER,
    map_width     INTEGER,
    map_height    INTEGER,
    fetched_at    INTEGER,
    wire_data     BLOB,        -- canonical fetched-replay payload: gzip(json(wire-shape array)). See replay_collector/wire.py.
    -- timestamps
    created_at    INTEGER NOT NULL DEFAULT {_NOW_MS},
    updated_at    INTEGER NOT NULL DEFAULT {_NOW_MS}
    -- Existing DBs may still carry a `raw_deprecated` BLOB column (renamed
    -- from the legacy `raw` by migrations/002_rename_raw_deprecated.py).
    -- Fresh DBs created from this _SCHEMA do not include it — slated for
    -- cleanup in a follow-up migration once the migrated DB has soaked.
);

-- This trigger fires a second UPDATE on every save. On the replays table
-- that means rewriting the row's blob overflow chain twice per Pass-2 save.
-- If ingest throughput becomes a bottleneck, drop this trigger and set
-- updated_at inline in save_full_data's UPDATE statement.
CREATE TRIGGER IF NOT EXISTS trg_replays_updated_at
AFTER UPDATE ON replays
BEGIN
    UPDATE replays SET updated_at = {_NOW_MS} WHERE id = NEW.id;
END;

CREATE TABLE IF NOT EXISTS replay_players (
    replay_id    TEXT NOT NULL REFERENCES replays(id) ON DELETE CASCADE,
    position     INTEGER NOT NULL,
    player_id    INTEGER NOT NULL REFERENCES players(id),
    current_name TEXT,
    stars        INTEGER,
    kills        INTEGER,
    PRIMARY KEY (replay_id, position)
);

CREATE INDEX IF NOT EXISTS idx_replay_players_player_id    ON replay_players(player_id);
CREATE INDEX IF NOT EXISTS idx_replay_players_current_name ON replay_players(current_name);
CREATE INDEX IF NOT EXISTS idx_replays_started             ON replays(started);
CREATE INDEX IF NOT EXISTS idx_replays_type                ON replays(type);
CREATE INDEX IF NOT EXISTS idx_replays_ladder_id           ON replays(ladder_id);

-- If the Pass-2 work-set query slows as the pending backlog grows, add:
--   CREATE INDEX IF NOT EXISTS idx_replays_pending_ffa
--     ON replays(started DESC, id)
--     WHERE ladder_id = 'ffa' AND wire_data IS NULL;
-- The partial predicate shrinks the index as rows fill, keeping it
-- proportional to the pending work set rather than the full FFA count.
"""

_conn: sqlite3.Connection | None = None


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH)
        # WAL lets readers and writers coexist (only writer-vs-writer
        # serializes), so a long-running fetch-gior doesn't block ad-hoc
        # query scripts. Persistent property of the DB file.
        _conn.execute("PRAGMA journal_mode = WAL;")
        _conn.execute("PRAGMA foreign_keys = ON;")
        # 64 MB page cache (vs SQLite's 2 MB default). Negative value = KB.
        _conn.execute("PRAGMA cache_size = -65536;")
        # Wait up to 5s on writer contention instead of failing immediately.
        # Already the Python sqlite3 default, but pin it explicitly here.
        _conn.execute("PRAGMA busy_timeout = 5000;")
        _conn.executescript(_SCHEMA)
    return _conn


# Ensure the DB closes on shutdown so SQLite removes its -wal/-shm sidecar
# files. atexit covers normal exits and Ctrl-C (SIGINT). SIGTERM bypasses
# Python's finalization by default; route it through sys.exit so atexit
# fires for `kill`, IDE stop buttons, and container shutdown too.
def _close_conn() -> None:
    global _conn
    if _conn is not None:
        # Refresh stale planner stats before close (no-op if nothing drifted).
        _conn.execute("PRAGMA optimize;")
        _conn.close()
        _conn = None


atexit.register(_close_conn)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


def has_full_data(replay_id: str) -> bool:
    """True iff we've fetched and stored the .gior payload for this replay."""
    cur = get_conn().execute(
        "SELECT 1 FROM replays WHERE id = ? AND wire_data IS NOT NULL LIMIT 1",
        (replay_id,),
    )
    return cur.fetchone() is not None


def cached_full_replay_stats(
    player_name: str,
) -> tuple[int, int | None, int | None]:
    """Return (count, min_started, max_started) of replays where `player_name`
    appears in the ranking and the .gior payload is stored."""
    # NOTE: keys on `players.name`. If a player has been renamed, replays under
    # the old name live in a different `players` row and won't be counted here.
    row = get_conn().execute(
        """
        SELECT COUNT(r.id), MIN(r.started), MAX(r.started)
        FROM replays r
        JOIN replay_players rp ON rp.replay_id = r.id
        JOIN players p          ON p.id = rp.player_id
        WHERE p.name = ? AND r.wire_data IS NOT NULL
        """,
        (player_name,),
    ).fetchone()
    return row[0], row[1], row[2]


def replay_counts_by_player(
    player_names: list[str],
) -> list[tuple[str, int, int, int]]:
    """For each name, return (name, total_listings, total_ffa, ffa_metadata_only)
    where `metadata_only` is FFA listings with no `wire_data` yet — i.e. the
    Pass 2 (.gior fetch) backlog. Cumulative across all runs. Ordered by
    backlog size, descending. Names with no replay data are absent from the
    result."""
    if not player_names:
        return []
    placeholders = ",".join("?" * len(player_names))
    return get_conn().execute(
        f"""
        SELECT p.name,
               COUNT(*) AS total_listings,
               SUM(CASE WHEN r.ladder_id = 'ffa' THEN 1 ELSE 0 END) AS total_ffa,
               SUM(CASE WHEN r.ladder_id = 'ffa' AND r.wire_data IS NULL THEN 1 ELSE 0 END) AS metadata_only
        FROM replays r
        JOIN replay_players rp ON rp.replay_id = r.id
        JOIN players p          ON p.id = rp.player_id
        WHERE p.name IN ({placeholders})
        GROUP BY p.name
        ORDER BY metadata_only DESC, p.name
        """,
        player_names,
    ).fetchall()


def pending_full_data_count(player_filter: list[str] | None = None) -> int:
    """Total replays in the Pass 2 backlog under the given filter. Counts
    distinct replays even if multiple filter-listed players are in the same
    game's ranking."""
    where = ["r.ladder_id = 'ffa'", "r.wire_data IS NULL"]
    params: list = []
    if player_filter:
        placeholders = ",".join("?" * len(player_filter))
        where.append(f"p.name IN ({placeholders})")
        params.extend(player_filter)

    sql = f"""
        SELECT COUNT(DISTINCT r.id)
        FROM replays r
        JOIN replay_players rp ON rp.replay_id = r.id
        JOIN players p          ON p.id = rp.player_id
        WHERE {" AND ".join(where)}
    """
    return get_conn().execute(sql, params).fetchone()[0]


def pending_full_data_work_set(
    player_filter: list[str] | None = None,
    limit: int | None = None,
) -> list[tuple[str, str, int]]:
    """Return rows of (replay_id, owner_name, started) for FFA replays whose
    .gior bytes haven't been fetched yet — i.e. the Pass 2 backlog.

    Each replay is "owned" by `MIN(p.name)` over its ranking (alphabetically
    first), which is the queue-head used for round-robin ordering. Output is
    sorted so that fetches alternate across owners, taking each owner's newest
    pending replay first: round 1 covers every owner's newest, round 2 their
    second-newest, and so on.

    `player_filter` (optional) restricts the work set to replays where at least
    one named player appears in the ranking, AND restricts ownership to those
    same players — so the round-robin balances among the listed players.

    Replays with no listing for any filter-listed player are excluded entirely."""
    where = ["r.ladder_id = 'ffa'", "r.wire_data IS NULL"]
    params: list = []
    if player_filter:
        placeholders = ",".join("?" * len(player_filter))
        where.append(f"p.name IN ({placeholders})")
        params.extend(player_filter)

    sql = f"""
        WITH owner AS (
            SELECT r.id     AS replay_id,
                   r.started,
                   MIN(p.name) AS owner_name
            FROM replays r
            JOIN replay_players rp ON rp.replay_id = r.id
            JOIN players p          ON p.id = rp.player_id
            WHERE {" AND ".join(where)}
            GROUP BY r.id, r.started
        )
        SELECT replay_id, owner_name, started
        FROM owner
        -- replay_id tiebreaks same-`started` rows so the LIMIT cutoff is
        -- stable across runs.
        ORDER BY ROW_NUMBER() OVER (
                     PARTITION BY owner_name ORDER BY started DESC, replay_id
                 ),
                 owner_name
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    return get_conn().execute(sql, params).fetchall()


def _player_id(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute("SELECT id FROM players WHERE name = ?", (name,)).fetchone()
    if row is not None:
        return row[0]
    cur = conn.execute("INSERT INTO players (name) VALUES (?)", (name,))
    return cur.lastrowid


def upsert_listing(entry: dict) -> bool:
    """Insert a listing-derived replay row + its ranking junction rows.
    No-op if a row with this id already exists. Returns True if inserted."""
    replay_id = entry["id"]
    conn = get_conn()
    with conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO replays
                (id, type, ladder_id, started, turns, player_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                replay_id,
                entry["type"],
                entry.get("ladder_id"),
                entry["started"],
                entry["turns"],
                len(entry.get("ranking", [])),
            ),
        )
        if cur.rowcount == 0:
            return False
        for pos, p in enumerate(entry.get("ranking", [])):
            pid = _player_id(conn, p["name"])
            conn.execute(
                """
                INSERT INTO replay_players
                    (replay_id, position, player_id, current_name, stars, kills)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    replay_id,
                    pos,
                    pid,
                    p.get("currentName"),
                    p.get("stars"),
                    p.get("kills"),
                ),
            )
        return True


def save_full_data(replay_id: str, decoded: list) -> None:
    """Update an existing replay row with the gzip+JSON-encoded `wire_data`
    payload + decoded fields. Requires the listing row to exist (call
    upsert_listing first)."""
    conn = get_conn()
    with conn:
        cur = conn.execute(
            """
            UPDATE replays
            SET version = ?, map_width = ?, map_height = ?,
                fetched_at = ?, wire_data = ?
            WHERE id = ?
            """,
            (
                decoded[0],
                decoded[2],
                decoded[3],
                int(time.time() * 1000),
                wire.encode(decoded),
                replay_id,
            ),
        )
        if cur.rowcount == 0:
            raise ValueError(
                f"no listing row for {replay_id}; call upsert_listing first"
            )
