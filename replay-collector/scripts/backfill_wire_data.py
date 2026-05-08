"""Phase 2 / migration helper — backfill `wire_data` from legacy `raw`.

For every row where `raw IS NOT NULL AND wire_data IS NULL`, decompress the
lz-string `raw` blob through the legacy `decompress_gior` path and re-encode
the wire-shape array as gzip+JSON via `replay_collector.wire.encode`.

Idempotent: skips rows that already have `wire_data` populated. Safe to
re-run if interrupted.

Run with (after migrations/001_add_wire_data.py has been applied):
    cd replay-collector/
    uv run python scripts/backfill_wire_data.py

Workers: 8 by default. Estimated ~5-7 min on M1 for 135k corpus.

A final spot-check verifies round-trip equality on 20 random backfilled rows
(`json.loads(json.dumps(decompress_gior(raw))) == wire.decode(wire_data)` —
the `json.dumps`/`loads` normalization collapses the unpaired-surrogate
representation that `decompress_gior` produces for non-BMP characters in
usernames; without it, raw `==` reports false positives).
"""

import argparse
import json
import multiprocessing as mp
import sqlite3
import sys
import time
from pathlib import Path

from replay_collector import wire
from replay_collector.generals_api import decompress_gior

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "generals.sqlite"
DEFAULT_WORKERS = 8
READ_BATCH = 500          # rows per fetchmany
WRITE_BATCH = 500         # UPDATEs per commit
POOL_CHUNKSIZE = 16       # items dispatched per pool task


def encode_one(row: tuple[str, bytes]) -> tuple[str, bytes]:
    """Worker: lz-string decode → gzip+JSON re-encode."""
    replay_id, raw = row
    return replay_id, wire.encode(decompress_gior(raw))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total rows processed (smoke testing)")
    args = ap.parse_args()

    if not DB_PATH.exists():
        sys.exit(f"DB not found: {DB_PATH}")

    # Two connections: one for streaming SELECTs, one for batched UPDATEs.
    # Keeps the read cursor's snapshot independent of write commits.
    read_conn = sqlite3.connect(DB_PATH)
    write_conn = sqlite3.connect(DB_PATH)
    for c in (read_conn, write_conn):
        c.execute("PRAGMA journal_mode = WAL;")

    # Sanity check: the column must already exist (migration 001 applied).
    cols = {r[1] for r in read_conn.execute("PRAGMA table_info(replays)").fetchall()}
    if "wire_data" not in cols:
        sys.exit("`wire_data` column missing — run migrations/001_add_wire_data.py first.")

    total = read_conn.execute(
        "SELECT COUNT(*) FROM replays WHERE raw IS NOT NULL AND wire_data IS NULL"
    ).fetchone()[0]
    if args.limit is not None:
        total = min(total, args.limit)
    print(f"Backfilling {total} replays with {args.workers} workers.")
    if total == 0:
        return

    sql = (
        "SELECT id, raw FROM replays "
        "WHERE raw IS NOT NULL AND wire_data IS NULL "
        "ORDER BY id"
    )
    if args.limit is not None:
        sql += f" LIMIT {args.limit}"

    def stream():
        cur = read_conn.execute(sql)
        while True:
            batch = cur.fetchmany(READ_BATCH)
            if not batch:
                return
            yield from batch

    pending: list[tuple[str, bytes]] = []
    processed = 0
    t0 = time.perf_counter()

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        write_conn.executemany(
            "UPDATE replays SET wire_data = ? WHERE id = ?",
            [(blob, rid) for rid, blob in pending],
        )
        write_conn.commit()
        pending = []

    with mp.Pool(args.workers) as pool:
        for result in pool.imap_unordered(encode_one, stream(), chunksize=POOL_CHUNKSIZE):
            pending.append(result)
            processed += 1
            if len(pending) >= WRITE_BATCH:
                flush()
                elapsed = time.perf_counter() - t0
                rate = processed / elapsed
                eta_min = (total - processed) / rate / 60 if rate > 0 else 0.0
                print(f"  {processed}/{total}  ({rate:.0f}/s, ETA {eta_min:.1f} min)")
        flush()

    elapsed = time.perf_counter() - t0
    print(f"\nProcessed {processed} replays in {elapsed:.1f}s ({processed/elapsed:.0f}/s)")

    # Verification 1: nothing left to do (skipped under --limit, since
    # we deliberately processed only a subset).
    if args.limit is None:
        leftover = read_conn.execute(
            "SELECT COUNT(*) FROM replays WHERE raw IS NOT NULL AND wire_data IS NULL"
        ).fetchone()[0]
        if leftover != 0:
            sys.exit(f"FAILED: {leftover} rows still missing wire_data")

    # Verification 2: spot-check 20 random backfilled rows for round-trip
    # equality. See module docstring for why we normalize via json round-trip
    # before comparing.
    print("Spot-checking 20 random backfilled rows for round-trip correctness...")
    failures = 0
    rows = read_conn.execute(
        "SELECT id, raw, wire_data FROM replays "
        "WHERE raw IS NOT NULL AND wire_data IS NOT NULL "
        "ORDER BY RANDOM() LIMIT 20"
    ).fetchall()
    for rid, raw, wd in rows:
        old_norm = json.loads(json.dumps(decompress_gior(raw)))
        new = wire.decode(wd)
        if old_norm != new:
            print(f"  MISMATCH: {rid}")
            failures += 1
    if failures:
        sys.exit(f"FAILED: {failures}/{len(rows)} spot-check mismatches")
    print(f"Spot-check passed ({len(rows)}/{len(rows)}).")


if __name__ == "__main__":
    main()
