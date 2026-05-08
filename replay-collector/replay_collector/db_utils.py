"""Lightweight DB helpers shared by migrations + one-shot scripts.

Kept separate from `db.py` so the latter can stay focused on the canonical
schema and the application connection-management lifecycle.
"""

import sqlite3


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return the set of column names present on `table`."""
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
