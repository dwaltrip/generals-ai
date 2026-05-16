"""Find the v30.9.2 deploy time by scanning listings around 2025-11-29.

The v30.9.2 patch (changelog dated 2025-11-29) added a kill/no-kill
partition to the bundle's `lbSort`. For each vanilla FFA v15+ replay in
the window, parses the wire data, computes both lbSort variants
(with/without `has_kill` partition), and classifies which rule the
server's listings encoded:

  OLD     listings match no-partition only — pre-v30.9.2 server behavior
  NEW     listings match partition only    — post-v30.9.2
  BOTH    both rules produce the same ranking (no discriminating power)
  NEITHER neither rule matches             — real sim / lbSort discrepancy
  SKIP    listings name doesn't byte-match wire usernames

Prints an hourly histogram across the window, then drills into the
transition hour minute-by-minute.

Usage (from replay-parser/):
    uv run python scripts/find_lbsort_deploy_time.py
"""
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
import sqlite3
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "replay-parser"))

from replay_parser._collector.config import DB_PATH
from replay_parser._collector.wire import decode as decode_blob
from replay_parser._shared import is_vanilla_ffa
from replay_parser.parser import parse_replay
from replay_parser.validator import deduce_ranking


# Window: 2 days centered on the changelog date.
WINDOW_START_MS = 1764288000000  # 2025-11-28 00:00 UTC
WINDOW_END_MS = 1764547200000    # 2025-12-01 00:00 UTC


def classify(state, replay, listings) -> str:
    usernames = replay.static.usernames
    try:
        listings_slots = [usernames.index(name) for (name,) in listings]
    except ValueError:
        return "SKIP"
    old_rank = deduce_ranking(state, partition_kill_no_kill=False)
    new_rank = deduce_ranking(state, partition_kill_no_kill=True)
    matches_old = listings_slots == old_rank
    matches_new = listings_slots == new_rank
    if matches_old and matches_new:
        return "BOTH"
    if matches_old:
        return "OLD"
    if matches_new:
        return "NEW"
    return "NEITHER"


def fmt_hour(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:00 UTC")


def fmt_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            """SELECT id, started, wire_data FROM replays
               WHERE type = 'classic'
                 AND version >= 15
                 AND wire_data IS NOT NULL
                 AND started >= ? AND started < ?
               ORDER BY started ASC""",
            (WINDOW_START_MS, WINDOW_END_MS),
        ).fetchall()

        results = []  # list of (started_ms, replay_id, classification)
        for replay_id, started, blob in rows:
            wire = decode_blob(blob)
            if not is_vanilla_ffa(wire):
                continue
            state, replay = parse_replay(blob)
            listings = conn.execute(
                """SELECT current_name FROM replay_players
                   WHERE replay_id = ? ORDER BY position""",
                (replay_id,),
            ).fetchall()
            label = classify(state, replay, listings)
            results.append((started, replay_id, label))
    finally:
        conn.close()

    print(f"Scanned {len(results)} vanilla v15+ FFA replays")
    print(f"  window: {fmt_ts(WINDOW_START_MS)} .. {fmt_ts(WINDOW_END_MS)}")
    print()

    # Hourly histogram
    hour_bucket = defaultdict(lambda: defaultdict(int))
    for started, _rid, label in results:
        hour_ms = (started // 3_600_000) * 3_600_000
        hour_bucket[hour_ms][label] += 1

    print("Hourly histogram:")
    print(f"  {'hour':22} {'OLD':>5} {'NEW':>5} {'BOTH':>5} {'NEITHER':>7} {'SKIP':>5}")
    # First NEW-dominant hour = where new strictly beats old AND there's a real signal.
    transition_hour = None
    for h_ms in sorted(hour_bucket):
        c = hour_bucket[h_ms]
        old = c.get("OLD", 0)
        new = c.get("NEW", 0)
        both = c.get("BOTH", 0)
        neither = c.get("NEITHER", 0)
        skip = c.get("SKIP", 0)
        marker = ""
        if transition_hour is None and new > old and new >= 3:
            transition_hour = h_ms
            marker = "  <-- first NEW-dominant hour"
        print(f"  {fmt_hour(h_ms):22} {old:>5} {new:>5} {both:>5} {neither:>7} {skip:>5}{marker}")

    if transition_hour is None:
        print("\nNo NEW-dominant hour found in window.")
        return

    print()
    print("Drill into transition window (the hour before + the transition hour):")
    drill_start = transition_hour - 3_600_000
    drill_end = transition_hour + 3_600_000
    for started, rid, label in results:
        if drill_start <= started < drill_end:
            print(f"  {fmt_ts(started)}  {rid:12}  {label}")


if __name__ == "__main__":
    main()
