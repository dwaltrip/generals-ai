# -------------------------------------------------------------------------------------
# NOTE: Apparently the website at RANKINGS_URL is broken for all seasons 41 and below.
#       They won't load in the browser either.
# -------------------------------------------------------------------------------------
"""Fetch generals.io season leaderboards and save each as season-N.json.

Iterates the inclusive range [start, end], skipping seasons whose output file
already exists. Output goes to data/leaderboards/.

Usage:
    uv run python scripts/fetch_season_leaderboards.py 1 42
    uv run python scripts/fetch_season_leaderboards.py 42
"""

import argparse
import json
from pathlib import Path
import sys
import time

from replay_collector.leaderboard import fetch_season_state


OUTPUT_DIR = Path("data/leaderboards")
SLEEP_SECONDS = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("start", type=int, help="first season number (inclusive)")
    parser.add_argument(
        "end",
        type=int,
        nargs="?",
        help="last season number (inclusive); defaults to start for a single fetch",
    )
    args = parser.parse_args()

    if args.end is None:
        args.end = args.start
    if args.start > args.end:
        parser.error("start must be <= end")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for season in range(args.start, args.end + 1):
        out_path = OUTPUT_DIR / f"season-{season}.json"
        if out_path.exists():
            print(f"season {season}: already exists, skipping", file=sys.stderr)
            continue

        print(f"season {season}: fetching...", file=sys.stderr)
        state = fetch_season_state(season)
        with out_path.open("w") as f:
            json.dump(state, f, indent=2)
        print(f"season {season}: wrote {out_path}", file=sys.stderr)

        if season != args.end:
            time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
