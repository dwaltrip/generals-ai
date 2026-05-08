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

Architecture: all DB I/O on the main thread; the multiprocessing pool only
runs the CPU-bound `encode_one`. Each iteration fetches a batch via the
natural `wire_data IS NULL` filter (already-processed rows fall out of the
next iteration's result set), encodes in workers, writes back, commits.
"""

import argparse
import json
import multiprocessing as mp
import sys
import time

from replay_collector import wire
from replay_collector.db import create_conn
from replay_collector.db_utils import columns
from replay_collector.generals_api import decompress_gior

DEFAULT_WORKERS = 8
BATCH_SIZE = 500
SPOT_CHECK_N = 20


def encode_one(row: tuple[str, bytes]) -> tuple[str, bytes]:
    """Worker: lz-string decode → gzip+JSON re-encode."""
    replay_id, raw = row
    return replay_id, wire.encode(decompress_gior(raw))


def run_backfill(workers: int, limit: int | None) -> None:
    """Backfill `wire_data` for every row where `raw IS NOT NULL AND
    wire_data IS NULL`. Multiprocessing pool for the CPU-bound encode;
    all DB reads/writes happen on the main thread."""
    try:
        read_conn = create_conn()
        write_conn = create_conn()
    except FileNotFoundError as e:
        sys.exit(str(e))

    if "wire_data" not in columns(read_conn, "replays"):
        sys.exit("`wire_data` column missing — run migrations/001_add_wire_data.py first.")

    total = read_conn.execute(
        "SELECT COUNT(*) FROM replays WHERE raw IS NOT NULL AND wire_data IS NULL"
    ).fetchone()[0]
    if limit is not None:
        total = min(total, limit)
    print(f"Backfilling {total} replays with {workers} workers.")
    if total == 0:
        return

    select_sql = (
        "SELECT id, raw FROM replays "
        "WHERE raw IS NOT NULL AND wire_data IS NULL "
        "ORDER BY id LIMIT ?"
    )
    update_sql = "UPDATE replays SET wire_data = ? WHERE id = ?"

    processed = 0
    t0 = time.perf_counter()

    with mp.Pool(workers) as pool:
        while True:
            fetch_n = BATCH_SIZE
            if limit is not None:
                fetch_n = min(fetch_n, limit - processed)
                if fetch_n <= 0:
                    break

            batch = read_conn.execute(select_sql, (fetch_n,)).fetchall()
            if not batch:
                break

            results = pool.map(encode_one, batch)
            write_conn.executemany(update_sql, [(blob, rid) for rid, blob in results])
            write_conn.commit()

            processed += len(batch)
            elapsed = time.perf_counter() - t0
            rate = processed / elapsed
            eta_min = (total - processed) / rate / 60 if rate > 0 else 0.0
            print(f"  {processed}/{total}  ({rate:.0f}/s, ETA {eta_min:.1f} min)")

    elapsed = time.perf_counter() - t0
    print(f"\nProcessed {processed} replays in {elapsed:.1f}s ({processed/elapsed:.0f}/s)")


def verify_backfill(limit: int | None) -> None:
    """Post-backfill checks: every fetched-replay row has wire_data, and a
    random spot-check round-trips equal between the legacy decode and the
    new gzip+JSON path. Exits non-zero on failure.

    The leftover-count check is skipped under --limit since we deliberately
    processed only a subset.

    The spot-check normalizes the legacy side via `json.loads(json.dumps(...))`
    before comparing — `decompress_gior` produces unpaired surrogates for
    non-BMP characters in usernames, and a fresh JSON round-trip collapses
    them to the proper code point. Without this, raw `==` reports false
    positives on emoji-bearing usernames.
    """
    try:
        conn = create_conn()
    except FileNotFoundError as e:
        sys.exit(str(e))

    if limit is None:
        leftover = conn.execute(
            "SELECT COUNT(*) FROM replays WHERE raw IS NOT NULL AND wire_data IS NULL"
        ).fetchone()[0]
        if leftover != 0:
            sys.exit(f"FAILED: {leftover} rows still missing wire_data")

    print(f"Spot-checking {SPOT_CHECK_N} random backfilled rows for round-trip correctness...")
    rows = conn.execute(
        "SELECT id, raw, wire_data FROM replays "
        "WHERE raw IS NOT NULL AND wire_data IS NOT NULL "
        "ORDER BY RANDOM() LIMIT ?",
        (SPOT_CHECK_N,),
    ).fetchall()
    failures = 0
    for rid, raw, wd in rows:
        old_norm = json.loads(json.dumps(decompress_gior(raw)))
        new = wire.decode(wd)
        if old_norm != new:
            print(f"  MISMATCH: {rid}")
            failures += 1
    if failures:
        sys.exit(f"FAILED: {failures}/{len(rows)} spot-check mismatches")
    print(f"Spot-check passed ({len(rows)}/{len(rows)}).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total rows processed (smoke testing)")
    args = ap.parse_args()

    run_backfill(args.workers, args.limit)
    verify_backfill(args.limit)


if __name__ == "__main__":
    main()
