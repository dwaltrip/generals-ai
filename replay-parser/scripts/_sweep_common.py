"""Shared scaffolding for corpus-sweep scripts.

Used by both `sweep_match_rates.py` (ranking match-rate validation) and
`parity_sweep.py` (Rust-vs-Python sim parity validation). Each consumer
brings its own per-replay check, Bucket schema, and report writer; this
module provides the common bones: candidate fetch, week bucketing,
random sampling, and chunked blob/listings pre-fetch.
"""
from collections import defaultdict
from datetime import UTC, datetime, timedelta
import random
import sqlite3
import sys

from replay_parser._collector.sql_helpers import version_range, wire_data_filter


# replay id | started (epoch ms, UTC) | version num
type ReplayInfo = tuple[str, int, int]


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def week_start(started_ms: int) -> str:
    """ISO Monday key (UTC) for a started-epoch-ms timestamp."""
    dt = datetime.fromtimestamp(started_ms / 1000, tz=UTC)
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d")


def fetch_candidates(conn: sqlite3.Connection, *, min_version: int = 15) -> list[ReplayInfo]:
    """Standard FFA + wire-data candidate set, ordered by started ASC."""
    rows = conn.execute(
        f"""SELECT id, started, version FROM replays
           WHERE ladder_id = 'ffa'
             AND {version_range('replays', min_version=min_version)}
             AND {wire_data_filter('replays')}
           ORDER BY started ASC"""
    ).fetchall()
    return rows


def bucket_and_sample(
    candidates: list[ReplayInfo],
    *,
    per_bucket: int,
    seed: int,
    skip_fn=None,
) -> tuple[dict[str, list[ReplayInfo]], list[ReplayInfo]]:
    """Bucket candidates by ISO week, then random-sample up to `per_bucket` per
    bucket. `skip_fn(replay_info) -> bool` lets a caller exclude replays from
    the pool (e.g. the v30.9.2 ambiguity window for ranking sweeps).

    Returns (bucket_pool, sampled_sorted_by_started).
    """
    bucket_pool: dict[str, list[ReplayInfo]] = defaultdict(list)
    for info in candidates:
        if skip_fn is not None and skip_fn(info):
            continue
        bucket_pool[week_start(info[1])].append(info)

    rng = random.Random(seed)
    sampled: list[ReplayInfo] = []
    for pool in bucket_pool.values():
        if len(pool) <= per_bucket:
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, per_bucket))
    sampled.sort(key=lambda r: r[1])
    return bucket_pool, sampled


def fetch_blobs(
    conn: sqlite3.Connection, ids: list[str], *, chunk: int = 500
) -> dict[str, bytes]:
    """Chunked `SELECT id, wire_data FROM replays WHERE id IN (...)`."""
    blobs: dict[str, bytes] = {}
    for i in range(0, len(ids), chunk):
        batch = ids[i : i + chunk]
        placeholders = ",".join("?" * len(batch))
        for rid, blob in conn.execute(
            f"SELECT id, wire_data FROM replays WHERE id IN ({placeholders})", batch
        ):
            blobs[rid] = blob
    return blobs


def fetch_listings(
    conn: sqlite3.Connection, ids: list[str], *, chunk: int = 500
) -> dict[str, list[str]]:
    """Chunked replay_players names, ordered by position per replay."""
    listings: dict[str, list[str]] = defaultdict(list)
    for i in range(0, len(ids), chunk):
        batch = ids[i : i + chunk]
        placeholders = ",".join("?" * len(batch))
        for rid, name in conn.execute(
            f"SELECT replay_id, current_name FROM replay_players "
            f"WHERE replay_id IN ({placeholders}) ORDER BY replay_id, position",
            batch,
        ):
            listings[rid].append(name)
    return listings
