"""Sweep the corpus and report listings-vs-deduced-ranking match rates by week.

Runs the version-aware ranking comparison over every v15+ FFA replay with
wire_data, buckets by ISO week (Monday-aligned UTC), and writes a markdown
report to replay-parser/tmp/.

Listings ranking comes from `replay_players.position` (server-side).
Deduced ranking comes from our parser's `deduce_ranking_for_replay`, which
picks the pre/post v30.9.2 lbSort rule from `started`.

Usage (from replay-parser/):
    uv run python scripts/sweep_match_rates.py
"""
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import sqlite3
import sys

from tabulate import tabulate

from replay_parser._collector.config import DB_PATH
from replay_parser._collector.sql_helpers import version_range, wire_data_filter
from replay_parser._collector.wire import decode as decode_blob
from replay_parser._shared import is_vanilla_ffa
from replay_parser.errors import ArmyOverflowError
from replay_parser.parser import parse_replay
from replay_parser.validator import (
    PRE_V30_9_2_CUTOFF_MS,
    POST_V30_9_2_CUTOFF_MS,
    deduce_ranking_for_replay,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "tmp"
PROGRESS_EVERY = 100
# SAMPLE_PER_BUCKET = 100   # random replays per week-bucket
SAMPLE_PER_BUCKET = 50   # random replays per week-bucket
RANDOM_SEED = 42
MAX_ERROR_SAMPLES = 20
MAX_MISS_IDS_PER_BUCKET = 3


# replay id | started (epoch ms, UTC) | version num
type ReplayInfo = tuple[str, int, int]


@dataclass
class Bucket:
    total: int = 0
    match: int = 0
    miss: int = 0
    nameskip: int = 0
    overflow: int = 0
    parse_err: int = 0
    nonvanilla: int = 0
    versions: set[int] = field(default_factory=set)
    miss_ids: list[str] = field(default_factory=list)


@dataclass
class SweepData:
    total_candidates: int
    sampled_replays: list[ReplayInfo]
    blobs_by_id: dict[str, bytes]
    listings_by_id: dict[str, list[str]]



def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def week_start(started_ms: int) -> str:
    dt = datetime.fromtimestamp(started_ms / 1000, tz=timezone.utc)
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d")


def get_sweep_data() -> SweepData:
    conn = sqlite3.connect(DB_PATH)
    try:
        log("Fetching candidate metadata...")
        # ---------------------------------------------------------------------
        # TODO: Look into restricting to player_count range: [4, 8]
        # Could this explain some of the non-matches we see?
        # I think the sweep is currently not handling `player_count`...
        # ---------------------------------------------------------------------
        candidates = conn.execute(
            f"""SELECT id, started, version FROM replays
               WHERE ladder_id = 'ffa'
                 AND {version_range('replays', min_version=15)}
                 AND {wire_data_filter('replays')}
               ORDER BY started ASC"""
        ).fetchall()
        total_candidates = len(candidates)
        log(f"  {total_candidates:,} candidate replays")

        # Bucket candidates by week, then random-sample SAMPLE_PER_BUCKET per
        # bucket. We do vanilla-FFA filtering AFTER sampling (decoding the wire
        # is part of the per-replay cost we want to bound).
        bucket_pool: dict[str, list[ReplayInfo]] = defaultdict(list)
        for replay_id, started, version in candidates:
            if PRE_V30_9_2_CUTOFF_MS < started < POST_V30_9_2_CUTOFF_MS:
                continue
            wk = week_start(started)
            bucket_pool[wk].append((replay_id, started, version))

        sampled: list[ReplayInfo] = []
        rng = random.Random(RANDOM_SEED)
        for wk, pool in bucket_pool.items():
            if len(pool) <= SAMPLE_PER_BUCKET:
                sampled.extend(pool)
            else:
                sampled.extend(rng.sample(pool, SAMPLE_PER_BUCKET))

        sampled.sort(key=lambda r: r[1])
        num_weeks = len(bucket_pool)
        log(f"  sampled {len(sampled):,} replays across {num_weeks} weeks")

        # Pre-fetch blobs + listings only for sampled IDs. We chunk the IN clause
        # to stay under SQLite's parameter limit.
        sampled_ids = [r[0] for r in sampled]
        blobs: dict[str, bytes] = {}
        listings_by_id: dict[str, list[str]] = defaultdict(list)
        CHUNK = 500
        log("Fetching blobs + listings for sampled replays...")
        for i in range(0, len(sampled_ids), CHUNK):
            chunk = sampled_ids[i:i+CHUNK]
            placeholders = ",".join("?" * len(chunk))
            for rid, blob in conn.execute(
                f"SELECT id, wire_data FROM replays WHERE id IN ({placeholders})", chunk,
            ):
                blobs[rid] = blob
            for rid, name in conn.execute(
                f"SELECT replay_id, current_name FROM replay_players "
                f"WHERE replay_id IN ({placeholders}) ORDER BY replay_id, position",
                chunk,
            ):
                listings_by_id[rid].append(name)
        log(f"  fetched {len(blobs):,} blobs")
    finally:
        conn.close()

    return SweepData(
        total_candidates=total_candidates,
        sampled_replays=sampled,
        blobs_by_id=blobs,
        listings_by_id=listings_by_id,
    )

def write_report(
    buckets: dict[str, Bucket],
    *,
    sampled_count: int,
    total_candidates: int,
    nonvanilla: int,
    parse_error_samples: list[tuple[str, str, str]],
    out_dir: Path,
) -> Path:
    """Build the markdown report and write it. Returns the output path."""
    table_rows = []
    for wk in sorted(buckets):
        b = buckets[wk]
        denom = b.match + b.miss
        pct = f"{100 * b.match / denom:.1f}%" if denom else "-"
        versions = "{" + ",".join(str(v) for v in sorted(b.versions)) + "}"
        sample = " ".join(b.miss_ids)
        table_rows.append([
            wk,
            b.total,
            b.match,
            b.miss,
            b.nameskip,
            b.overflow,
            b.parse_err,
            b.nonvanilla,
            pct,
            versions,
            sample,
        ])

    table = tabulate(
        table_rows,
        headers=[
            "week (mon)",
            "total",
            "match",
            "miss",
            "name-skip",
            "overflow",
            "parse-err",
            "non-vanilla",
            "%match",
            "versions",
            "sample mismatch ids",
        ],
        tablefmt="github",
    )

    now = datetime.now(tz=timezone.utc)
    lines = [
        "# Sweep: listings vs deduced ranking match rates",
        "",
        f"Generated: {now.isoformat()}",
        f"Total v15+ classic candidates: {total_candidates:,}",
        f"Sample target: {SAMPLE_PER_BUCKET}/week (random, seed={RANDOM_SEED})",
        f"Sampled (after gap-skip): {sampled_count:,} across {len(buckets)} buckets",
        f"Non-vanilla in sample: {nonvanilla:,}",
        "",
        "%match denominator = match + miss (excludes name-skip, overflow, parse-err, non-vanilla).",
        "",
        table,
        "",
    ]
    if parse_error_samples:
        lines.extend([
            "",
            f"## Parse error samples (first {len(parse_error_samples)})",
            "",
        ])
        for rid, err_type, msg in parse_error_samples:
            lines.append(f"- `{rid}`  {err_type}: {msg}")
        lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sweep_match_rates-{now.strftime('%Y%m%d-%H%M')}.md"
    out_path.write_text("\n".join(lines))
    return out_path


def main():
    data = get_sweep_data()
    total = len(data.sampled_replays)

    buckets: dict[str, Bucket] = defaultdict(Bucket)
    parse_error_samples: list[tuple[str, str, str]] = []  # (id, type, msg)

    nonvanilla = 0
    processed = 0

    for replay_id, started, version in data.sampled_replays:
        processed += 1
        if processed % PROGRESS_EVERY == 0:
            log(f"  ... {processed:,}/{total:,}")

        def handle_parse_error(bucket, label, e):
            bucket.parse_err += 1
            if len(parse_error_samples) < MAX_ERROR_SAMPLES:
                parse_error_samples.append((replay_id, type(e).__name__, str(e)))
            log(f"  {label} {replay_id}: {type(e).__name__}: {e}")

        b = buckets[week_start(started)]
        b.total += 1
        b.versions.add(version)

        blob = data.blobs_by_id[replay_id]
        try:
            wire = decode_blob(blob)
        except Exception as e:
            handle_parse_error(b, 'decode error', e)
            continue

        if not is_vanilla_ffa(wire):
            nonvanilla += 1
            b.nonvanilla += 1
            continue

        try:
            state, replay = parse_replay(blob)
        except ArmyOverflowError:
            b.overflow += 1
            continue
        except Exception as e:
            handle_parse_error(b, 'parse error', e)
            continue

        listings_names = data.listings_by_id.get(replay_id, [])
        usernames = replay.static.usernames
        try:
            listings_slots = [usernames.index(name) for name in listings_names]
        except ValueError:
            b.nameskip += 1
            continue

        deduced = deduce_ranking_for_replay(state, started)
        if listings_slots == deduced:
            b.match += 1
        else:
            b.miss += 1
            if len(b.miss_ids) < MAX_MISS_IDS_PER_BUCKET:
                b.miss_ids.append(replay_id)

    out_path = write_report(
        buckets,
        sampled_count=total,
        total_candidates=data.total_candidates,
        nonvanilla=nonvanilla,
        parse_error_samples=parse_error_samples,
        out_dir=OUT_DIR,
    )

    log(f"Wrote: {out_path}")
    log(f"Total candidates: {data.total_candidates:,}")
    log(f"Sampled: {total:,}")
    log(f"  non-vanilla: {nonvanilla:,}")
    log(f"  buckets: {len(buckets)}")


if __name__ == "__main__":
    main()
