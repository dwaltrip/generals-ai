"""Compute distribution stats on game length (turns) for a slice of replays.

Usage (from replay-collector/):
    uv run python scripts/analyze_turn_stats.py
    uv run python scripts/analyze_turn_stats.py --type classic --player-count 8
    uv run python scripts/analyze_turn_stats.py --type bigteam --player-count 8 --require-wire-data

Defaults target the dominant 8-player FFA bucket (type=classic, player_count=8).
"""

import argparse
import statistics
import sys

from replay_collector.db import create_conn


PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def fetch_turns(replay_type: str, player_count: int, require_wire_data: bool) -> list[int]:
    sql = "SELECT turns FROM replays WHERE type = ? AND player_count = ?"
    params: list = [replay_type, player_count]
    if require_wire_data:
        sql += " AND wire_data IS NOT NULL"
    try:
        conn = create_conn()
    except FileNotFoundError as e:
        sys.exit(str(e))
    rows = conn.execute(sql, params).fetchall()
    return [r[0] for r in rows]


def percentile(sorted_values: list[int], pct: float) -> int:
    """Same convention as scripts/analyze_stars_decay.py: int(pct * n) into a sorted list."""
    n = len(sorted_values)
    idx = min(int(pct * n), n - 1)
    return sorted_values[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--type", dest="replay_type", default="classic")
    parser.add_argument("--player-count", type=int, default=8)
    parser.add_argument(
        "--require-wire-data",
        action="store_true",
        help="Only include replays whose .gior payload has been fetched",
    )
    args = parser.parse_args()

    turns = fetch_turns(args.replay_type, args.player_count, args.require_wire_data)
    n = len(turns)
    slice_desc = (
        f"type={args.replay_type} player_count={args.player_count}"
        f"{' wire-data-only' if args.require_wire_data else ''}"
    )
    print(f"slice: {slice_desc}  n={n}", file=sys.stderr)
    if not n:
        return

    turns.sort()
    mean = statistics.fmean(turns)
    stdev = statistics.pstdev(turns) if n > 1 else 0.0
    print(
        f"mean={mean:.1f}  stdev={stdev:.1f}  min={turns[0]}  max={turns[-1]}"
    )
    print("percentiles:")
    for p in PERCENTILES:
        v = percentile(turns, p / 100)
        print(f"  p{p:<2} = {v}")


if __name__ == "__main__":
    main()
