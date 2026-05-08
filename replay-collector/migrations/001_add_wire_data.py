"""Migration 001 — add `wire_data` BLOB column to `replays`.

Idempotent: checks PRAGMA table_info before ALTER. Safe to re-run.

Run with:
    cd replay-collector/
    uv run python migrations/001_add_wire_data.py
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "generals.sqlite"


def main() -> None:
    if not DB_PATH.exists():
        sys.exit(f"DB not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(replays)").fetchall()}
        if "wire_data" in cols:
            print("`wire_data` column already present in `replays` — nothing to do.")
            return
        conn.execute("ALTER TABLE replays ADD COLUMN wire_data BLOB;")
        conn.commit()
        print("Added `wire_data` BLOB column to `replays`.")


if __name__ == "__main__":
    main()
