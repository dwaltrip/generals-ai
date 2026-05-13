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
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tabulate import tabulate

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "replay-parser"))

from replay_parser._collector.config import DB_PATH
from replay_parser._collector.wire import decode as decode_blob
from replay_parser._shared import is_vanilla_ffa
from replay_parser.errors import ArmyOverflowError
from replay_parser.parser import parse_replay
from replay_parser.validator import (
    PRE_V30_9_2_CUTOFF_MS,
    POST_V30_9_2_CUTOFF_MS,
    deduce_ranking_for_replay,
)

OUT_DIR = REPO_ROOT / "replay-parser" / "tmp"
PROGRESS_EVERY = 1000
SAMPLE_PER_BUCKET = 100   # random replays per week-bucket
RANDOM_SEED = 42


def week_start(started_ms: int) -> str:
    dt = datetime.fromtimestamp(started_ms / 1000, tz=timezone.utc)
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        print("Fetching candidate metadata...", file=sys.stderr)
        candidates = conn.execute(
            """SELECT id, started, version FROM replays
               WHERE type = 'classic'
                 AND version >= 15
                 AND wire_data IS NOT NULL
               ORDER BY started ASC"""
        ).fetchall()
        total_candidates = len(candidates)
        print(f"  {total_candidates:,} candidate replays", file=sys.stderr)

        # Bucket candidates by week, then random-sample SAMPLE_PER_BUCKET per
        # bucket. We do vanilla-FFA filtering AFTER sampling (decoding the wire
        # is part of the per-replay cost we want to bound).
        rng = random.Random(RANDOM_SEED)
        bucket_pool: dict[str, list[tuple]] = defaultdict(list)
        for replay_id, started, version in candidates:
            if PRE_V30_9_2_CUTOFF_MS < started < POST_V30_9_2_CUTOFF_MS:
                continue
            wk = week_start(started)
            bucket_pool[wk].append((replay_id, started, version))

        sampled: list[tuple[str, int, int]] = []
        for wk, pool in bucket_pool.items():
            if len(pool) <= SAMPLE_PER_BUCKET:
                sampled.extend(pool)
            else:
                sampled.extend(rng.sample(pool, SAMPLE_PER_BUCKET))
        sampled.sort(key=lambda r: r[1])
        print(f"  sampled {len(sampled):,} replays across {len(bucket_pool)} buckets", file=sys.stderr)

        # Pre-fetch blobs + listings only for sampled IDs. We chunk the IN clause
        # to stay under SQLite's parameter limit.
        sampled_ids = [r[0] for r in sampled]
        blobs: dict[str, bytes] = {}
        listings_by_id: dict[str, list[str]] = defaultdict(list)
        CHUNK = 500
        print("Fetching blobs + listings for sampled replays...", file=sys.stderr)
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
        print(f"  fetched {len(blobs):,} blobs", file=sys.stderr)
    finally:
        conn.close()

    # Per-bucket counters
    bucket_total: dict[str, int] = defaultdict(int)
    bucket_match: dict[str, int] = defaultdict(int)
    bucket_miss: dict[str, int] = defaultdict(int)
    bucket_nameskip: dict[str, int] = defaultdict(int)
    bucket_overflow: dict[str, int] = defaultdict(int)
    bucket_parse_err: dict[str, int] = defaultdict(int)
    bucket_versions: dict[str, set[int]] = defaultdict(set)
    bucket_miss_ids: dict[str, list[str]] = defaultdict(list)
    parse_error_samples: list[tuple[str, str, str]] = []  # (id, type, msg)

    nonvanilla = 0
    processed = 0

    for replay_id, started, version in sampled:
        processed += 1
        if processed % PROGRESS_EVERY == 0:
            print(f"  ... {processed:,}/{len(sampled):,}", file=sys.stderr)

        blob = blobs[replay_id]
        try:
            wire = decode_blob(blob)
        except Exception as e:
            wk = week_start(started)
            bucket_total[wk] += 1
            bucket_versions[wk].add(version)
            bucket_parse_err[wk] += 1
            if len(parse_error_samples) < 20:
                parse_error_samples.append((replay_id, type(e).__name__, str(e)))
            print(f"  decode error {replay_id}: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        if not is_vanilla_ffa(wire):
            nonvanilla += 1
            continue

        wk = week_start(started)
        bucket_total[wk] += 1
        bucket_versions[wk].add(version)

        try:
            state, replay = parse_replay(blob)
        except ArmyOverflowError:
            bucket_overflow[wk] += 1
            continue
        except Exception as e:
            bucket_parse_err[wk] += 1
            if len(parse_error_samples) < 20:
                parse_error_samples.append((replay_id, type(e).__name__, str(e)))
            print(f"  parse error {replay_id}: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        listings_names = listings_by_id.get(replay_id, [])
        usernames = replay.static.usernames
        try:
            listings_slots = [usernames.index(name) for name in listings_names]
        except ValueError:
            bucket_nameskip[wk] += 1
            continue

        deduced = deduce_ranking_for_replay(state, started)
        if listings_slots == deduced:
            bucket_match[wk] += 1
        else:
            bucket_miss[wk] += 1
            if len(bucket_miss_ids[wk]) < 3:
                bucket_miss_ids[wk].append(replay_id)

    weeks = sorted(bucket_total)
    table_rows = []
    for wk in weeks:
        match = bucket_match[wk]
        miss = bucket_miss[wk]
        nameskip = bucket_nameskip[wk]
        overflow = bucket_overflow[wk]
        parse_err = bucket_parse_err[wk]
        denom = match + miss
        pct = f"{100 * match / denom:.1f}%" if denom else "-"
        versions = "{" + ",".join(str(v) for v in sorted(bucket_versions[wk])) + "}"
        sample = " ".join(bucket_miss_ids[wk])
        table_rows.append([wk, bucket_total[wk], match, miss, nameskip, overflow, parse_err, pct, versions, sample])

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
            "%match",
            "versions",
            "sample mismatch ids",
        ],
        tablefmt="github",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M")
    out_path = OUT_DIR / f"sweep_match_rates-{now}.md"

    header_lines = [
        f"# Sweep: listings vs deduced ranking match rates",
        f"",
        f"Generated: {datetime.now(tz=timezone.utc).isoformat()}",
        f"Total v15+ classic candidates: {total_candidates:,}",
        f"Sample target: {SAMPLE_PER_BUCKET}/week (random, seed={RANDOM_SEED})",
        f"Sampled (after gap-skip): {len(sampled):,} across {len(bucket_pool)} buckets",
        f"Non-vanilla in sample (filtered out): {nonvanilla:,}",
        f"",
        f"%match denominator = match + miss (excludes name-skip, overflow, parse-err).",
        f"",
        table,
        "",
    ]
    if parse_error_samples:
        header_lines.extend([
            "",
            f"## Parse error samples (first {len(parse_error_samples)})",
            "",
        ])
        for rid, err_type, msg in parse_error_samples:
            header_lines.append(f"- `{rid}`  {err_type}: {msg}")
        header_lines.append("")
    out_path.write_text("\n".join(header_lines))

    print(f"Wrote: {out_path}", file=sys.stderr)
    print(f"Total candidates: {total_candidates:,}", file=sys.stderr)
    print(f"Sampled: {len(sampled):,}", file=sys.stderr)
    print(f"  non-vanilla: {nonvanilla:,}", file=sys.stderr)
    print(f"  buckets: {len(weeks)}", file=sys.stderr)


if __name__ == "__main__":
    main()
