"""Migration 002 — rename `raw` -> `raw_deprecated`.

Run AFTER:
- 001_add_wire_data.py has been applied
- scripts/backfill_wire_data.py has been run to completion
- All reader code paths have switched from `raw` to `wire_data`
- Collector save path no longer writes to `raw` (this is the case as of
  the wire_data migration commit — db.save_full_data writes wire_data only)

Idempotent: refuses to run if backfill is incomplete or schema is already
in the renamed state. Safe to re-run.

Run with:
    cd replay-collector/
    uv run python migrations/002_rename_raw_deprecated.py
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
        cols = columns(conn, "replays")

        if "raw_deprecated" in cols and "raw" not in cols:
            print("Already renamed (raw_deprecated present, raw absent) — nothing to do.")
            return
        if "raw" not in cols:
            sys.exit("Cannot rename: `raw` column not found in `replays`.")
        if "raw_deprecated" in cols:
            sys.exit(
                "Cannot rename: both `raw` and `raw_deprecated` columns exist. "
                "Inspect the schema and resolve manually."
            )
        if "wire_data" not in cols:
            sys.exit("Cannot rename: `wire_data` column missing — migration 001 not applied.")

        # Safety gate: every fetched-replay row must have wire_data populated
        # before we sever raw from the active code path.
        leftover = conn.execute(
            "SELECT COUNT(*) FROM replays WHERE raw IS NOT NULL AND wire_data IS NULL"
        ).fetchone()[0]
        if leftover:
            sys.exit(
                f"Cannot rename: {leftover} rows have `raw` but no `wire_data`. "
                "Run scripts/backfill_wire_data.py first."
            )

        conn.execute("ALTER TABLE replays RENAME COLUMN raw TO raw_deprecated;")
    print("Renamed `raw` -> `raw_deprecated`.")


if __name__ == "__main__":
    main()
