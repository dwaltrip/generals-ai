"""List top-N players by replay count in our dataset.

Counts rows in `replay_players` per player (i.e. games where the player
appears in the ranking). Joins keep on `players.name`, so a renamed player
shows up under each name they've used.

Usage (from replay-collector/):
    uv run python scripts/players_by_game_count.py
    uv run python scripts/players_by_game_count.py -n 50
    uv run python scripts/players_by_game_count.py --require-wire-data
    uv run python scripts/players_by_game_count.py --csv out/top.csv
"""

import argparse
import csv
import sys
from pathlib import Path

from replay_collector.db import create_conn
from replay_collector.sql_helpers import ffa_match_filter, wire_data_filter


def fetch_by_game_count(top_n: int, require_wire_data: bool) -> list[tuple[str, int]]:
    where = list(filter(None, [
        ffa_match_filter("r"),
        (wire_data_filter("r") if require_wire_data else None),
    ]))
    if require_wire_data:
        where.append(wire_data_filter("r"))
    sql = f"""
        SELECT p.name, COUNT(*) AS games
        FROM replay_players rp
        JOIN players p ON p.id = rp.player_id
        JOIN replays r ON r.id = rp.replay_id
        WHERE {" AND ".join(where)}
        GROUP BY p.id ORDER BY games DESC, p.name LIMIT ?
    """

    try:
        conn = create_conn()
    except FileNotFoundError as e:
        sys.exit(str(e))
    return conn.execute(sql, (top_n,)).fetchall()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", type=int, default=20, help="number of players to show")
    parser.add_argument(
        "--require-wire-data",
        action="store_true",
        help="Only count replays whose .gior payload has been fetched",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        metavar="PATH",
        help="Write results as CSV to PATH (columns: name, games)",
    )
    args = parser.parse_args()

    rows = fetch_by_game_count(args.n, args.require_wire_data)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "games"])
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}", file=sys.stderr)
        return

    slice_desc = "wire-data-only" if args.require_wire_data else "all replays"
    print(f"top {len(rows)} players ({slice_desc}):", file=sys.stderr)
    name_w = max((len(name) for name, _ in rows), default=4)
    for rank, (name, games) in enumerate(rows, 1):
        print(f"{rank:>3}. {name:<{name_w}}  {games}")


if __name__ == "__main__":
    main()
