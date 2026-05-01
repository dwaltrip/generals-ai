import argparse
import logging
import sys
from pathlib import Path

from replay_collector.runner import (
    DEFAULT_MAX_FAILURES,
    DEFAULT_MAX_LISTINGS_PER_USER,
    collect_many,
)


def load_players(path: Path) -> list[str]:
    """One username per line. Strips whitespace and skips blanks; preserves
    internal spaces (generals.io usernames may contain them)."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(prog="replay_collector")
    parser.add_argument(
        "players_file", type=Path,
        help="text file with one generals.io username per line",
    )
    parser.add_argument(
        "--n-ffa", type=int, required=True,
        help="target number of FFA replays to fetch per player",
    )
    parser.add_argument(
        "--max-listings", type=int, default=DEFAULT_MAX_LISTINGS_PER_USER,
        help=f"max listings to walk per player (default: {DEFAULT_MAX_LISTINGS_PER_USER})",
    )
    parser.add_argument(
        "--max-failures", type=int, default=DEFAULT_MAX_FAILURES,
        help=f"abort the run after this many HTTP failures (default: {DEFAULT_MAX_FAILURES})",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    players = load_players(args.players_file)
    if not players:
        parser.error(f"no usernames found in {args.players_file}")

    run = collect_many(
        players,
        n_ffa=args.n_ffa,
        max_listings=args.max_listings,
        max_failures=args.max_failures,
    )
    sys.exit(1 if run.aborted else 0)


if __name__ == "__main__":
    main()
