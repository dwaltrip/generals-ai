"""Migration 001 — add `wire_data` BLOB column to `replays`.

Idempotent: checks PRAGMA table_info before ALTER. Safe to re-run.

Run with:
    cd replay-collector/
    uv run python migrations/001_add_wire_data.py
"""

import sys

from replay_collector.db import create_conn
from replay_collector.db_utils import columns


def main() -> None:
    try:
        conn = create_conn()
    except FileNotFoundError as e:
        sys.exit(str(e))

    with conn:
        if "wire_data" in columns(conn, "replays"):
            print("`wire_data` column already present in `replays` — nothing to do.")
            return
        conn.execute("ALTER TABLE replays ADD COLUMN wire_data BLOB;")
    print("Added `wire_data` BLOB column to `replays`.")


if __name__ == "__main__":
    main()
