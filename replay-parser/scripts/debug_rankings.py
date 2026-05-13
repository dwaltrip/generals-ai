"""Show DB-listings ranking vs simulator-deduced ranking for one or more replays.

Usage (from replay-parser/):
    uv run python scripts/show_ranking.py YC-ZjiMiY KVt1xVdFd
"""
import argparse
import sqlite3
from datetime import datetime, timezone

from tabulate import tabulate

from replay_parser._collector.config import DB_PATH
from replay_parser._collector.wire import decode as decode_blob
from replay_parser.errors import ArmyOverflowError
from replay_parser.parser import parse_replay
from replay_parser.validator import deduce_ranking_for_replay


def fetch(conn, replay_id):
    row = conn.execute(
        "SELECT started, version, wire_data FROM replays WHERE id = ?",
        (replay_id,),
    ).fetchone()
    if row is None:
        return None, None, None, None
    started, version, blob = row
    listings_names = [
        name for (name,) in conn.execute(
            "SELECT current_name FROM replay_players "
            "WHERE replay_id = ? ORDER BY position",
            (replay_id,),
        )
    ]
    return started, version, blob, listings_names


def render(replay_id, started, version, listings_names, deduced_names):
    when = datetime.fromtimestamp(started / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n = max(len(listings_names), len(deduced_names))
    rows = []
    for i in range(n):
        a = listings_names[i] if i < len(listings_names) else ""
        b = deduced_names[i] if i < len(deduced_names) else ""
        mark = "✓" if a and b and a == b else "✗"
        rows.append([i + 1, a, b, mark])
    header = f"=== {replay_id}  (v{version}, {when}, {n} players) ==="
    print(header)
    print(tabulate(rows, headers=["rank", "listings (DB)", "deduced (sim)", ""]))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay_ids", nargs="+")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        for replay_id in args.replay_ids:
            started, version, blob, listings_names = fetch(conn, replay_id)
            if blob is None:
                print(f"=== {replay_id} ===\n  NOT FOUND in DB\n")
                continue
            try:
                state, replay = parse_replay(blob)
            except ArmyOverflowError as e:
                print(f"=== {replay_id} ===\n  parse skipped: {e}\n")
                continue
            except Exception as e:
                print(f"=== {replay_id} ===\n  parse error: {type(e).__name__}: {e}\n")
                continue
            deduced_slots = deduce_ranking_for_replay(state, started)
            deduced_names = [replay.static.usernames[p] for p in deduced_slots]
            render(replay_id, started, version, listings_names, deduced_names)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
