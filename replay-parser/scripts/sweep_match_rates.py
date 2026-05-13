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
PROGRESS_EVERY = 5000


def week_start(started_ms: int) -> str:
    dt = datetime.fromtimestamp(started_ms / 1000, tz=timezone.utc)
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        print("Fetching replay list...", file=sys.stderr)
        replays = conn.execute(
            """SELECT id, started, version, wire_data FROM replays
               WHERE type = 'classic'
                 AND version >= 15
                 AND wire_data IS NOT NULL
               ORDER BY started ASC"""
        ).fetchall()
        print(f"  {len(replays):,} candidate replays", file=sys.stderr)

        print("Fetching listings...", file=sys.stderr)
        listings_by_id: dict[str, list[str]] = defaultdict(list)
        for replay_id, name in conn.execute(
            "SELECT replay_id, current_name FROM replay_players ORDER BY replay_id, position"
        ):
            listings_by_id[replay_id].append(name)
        print(f"  {len(listings_by_id):,} replays with listings", file=sys.stderr)
    finally:
        conn.close()

    # Per-bucket counters
    bucket_total: dict[str, int] = defaultdict(int)
    bucket_match: dict[str, int] = defaultdict(int)
    bucket_miss: dict[str, int] = defaultdict(int)
    bucket_nameskip: dict[str, int] = defaultdict(int)
    bucket_overflow: dict[str, int] = defaultdict(int)
    bucket_versions: dict[str, set[int]] = defaultdict(set)
    bucket_miss_ids: dict[str, list[str]] = defaultdict(list)

    in_gap = 0
    nonvanilla = 0
    processed = 0

    for replay_id, started, version, blob in replays:
        processed += 1
        if processed % PROGRESS_EVERY == 0:
            print(f"  ... {processed:,}/{len(replays):,}", file=sys.stderr)

        if PRE_V30_9_2_CUTOFF_MS < started < POST_V30_9_2_CUTOFF_MS:
            in_gap += 1
            continue

        wire = decode_blob(blob)
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
        denom = match + miss
        pct = f"{100 * match / denom:.1f}%" if denom else "-"
        versions = "{" + ",".join(str(v) for v in sorted(bucket_versions[wk])) + "}"
        sample = " ".join(bucket_miss_ids[wk])
        table_rows.append([wk, bucket_total[wk], match, miss, nameskip, overflow, pct, versions, sample])

    table = tabulate(
        table_rows,
        headers=["week (mon)", "total", "match", "miss", "name-skip", "overflow", "%match", "versions", "sample mismatch ids"],
        tablefmt="github",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M")
    out_path = OUT_DIR / f"sweep_match_rates-{now}.md"

    header_lines = [
        f"# Sweep: listings vs deduced ranking match rates",
        f"",
        f"Generated: {datetime.now(tz=timezone.utc).isoformat()}",
        f"Candidate v15+ classic replays scanned: {len(replays):,}",
        f"Non-vanilla (filtered out): {nonvanilla:,}",
        f"In v30.9.2 ambiguity gap (skipped): {in_gap}",
        f"",
        f"%match denominator = match + miss (excludes name-skip, overflow, in-gap).",
        f"",
        table,
        "",
    ]
    out_path.write_text("\n".join(header_lines))

    print(f"Wrote: {out_path}", file=sys.stderr)
    print(f"Total scanned: {len(replays):,}", file=sys.stderr)
    print(f"  non-vanilla: {nonvanilla:,}", file=sys.stderr)
    print(f"  in-gap: {in_gap}", file=sys.stderr)
    print(f"  buckets: {len(weeks)}", file=sys.stderr)


if __name__ == "__main__":
    main()
