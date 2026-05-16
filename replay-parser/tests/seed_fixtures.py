"""Seed wire.bin files for all fixtures from the collector DB.

One-time per fixture set. Pulls each fixture's wire_data blob and writes
it to `tests/fixtures/<name>/wire.bin`. Skips fixtures whose wire.bin
already exists.

Usage (from replay-parser/):
    uv run python tests/seed_fixtures.py
"""
import sqlite3
import sys

from replay_parser._collector.config import DB_PATH

from _fixture_lib import FIXTURES, FIXTURES_DIR


def main() -> int:
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(DB_PATH)
    try:
        n_ok = 0
        for spec in FIXTURES:
            fixture_dir = FIXTURES_DIR / spec.name
            fixture_dir.mkdir(parents=True, exist_ok=True)
            wire_path = fixture_dir / "wire.bin"
            if wire_path.exists():
                print(f"  SKIP {spec.name:<30} wire.bin exists")
                n_ok += 1
                continue
            row = conn.execute(
                "SELECT wire_data FROM replays WHERE id = ?", (spec.replay_id,)
            ).fetchone()
            if row is None or row[0] is None:
                print(f"  FAIL {spec.name:<30} no wire_data for rid={spec.replay_id}")
                continue
            wire_path.write_bytes(row[0])
            n_ok += 1
            print(f"  OK   {spec.name:<30} rid={spec.replay_id} bytes={len(row[0]):,}")
    finally:
        conn.close()

    print(f"\n{n_ok}/{len(FIXTURES)} fixtures seeded")
    return 0 if n_ok == len(FIXTURES) else 1


if __name__ == "__main__":
    sys.exit(main())
