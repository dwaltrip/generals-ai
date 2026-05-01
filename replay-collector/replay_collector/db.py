import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "generals.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS replays (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    ladder_id   TEXT,
    started     INTEGER NOT NULL,
    turns       INTEGER NOT NULL,
    version     INTEGER NOT NULL,
    map_width   INTEGER NOT NULL,
    map_height  INTEGER NOT NULL,
    fetched_at  INTEGER NOT NULL,
    raw         BLOB NOT NULL
    -- Future: decoded_flat TEXT (JSON of the wire-shape array). Lets us query
    -- into game state via SQLite JSON ops without decompressing in code.
    -- Adds ~3.3x storage (~5GB at 100k replays). Backfillable from `raw`.
);

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
"""

_conn: sqlite3.Connection | None = None


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH)
        _conn.execute("PRAGMA foreign_keys = ON;")
        _conn.executescript(_SCHEMA)
    return _conn


def has_replay(replay_id: str) -> bool:
    cur = get_conn().execute("SELECT 1 FROM replays WHERE id = ? LIMIT 1", (replay_id,))
    return cur.fetchone() is not None


def _player_id(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute("SELECT id FROM players WHERE name = ?", (name,)).fetchone()
    if row is not None:
        return row[0]
    cur = conn.execute("INSERT INTO players (name) VALUES (?)", (name,))
    return cur.lastrowid


def save_replay(metadata: dict, decoded: list, raw: bytes) -> bool:
    """Persist a replay with its ranking junction rows. Returns True if newly
    inserted, False if a row with this id already existed (no-op)."""
    replay_id = metadata["id"]
    conn = get_conn()
    with conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO replays
                (id, type, ladder_id, started, turns, version,
                 map_width, map_height, fetched_at, raw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                replay_id,
                metadata["type"],
                metadata.get("ladder_id"),
                metadata["started"],
                metadata["turns"],
                decoded[0],
                decoded[2],
                decoded[3],
                int(time.time() * 1000),
                raw,
            ),
        )
        if cur.rowcount == 0:
            return False
        for pos, entry in enumerate(metadata.get("ranking", [])):
            pid = _player_id(conn, entry["name"])
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
                    entry.get("currentName"),
                    entry.get("stars"),
                    entry.get("kills"),
                ),
            )
        return True
