"""Check which player names in a txt file exist on generals.io.

Reads one username per line and calls `generals_api.user_exists` for each.
Prints `<exists>\t<name>` to stdout, where <exists> is "yes" or "no".

Usage:
    uv run python scripts/check_users_exist.py data/players.txt
"""

import argparse
import sys
from pathlib import Path

from replay_collector import generals_api
from replay_collector.client import RateLimiter, TrackedClient, make_client
from replay_collector.runner import DEFAULT_RATES


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="txt file with one username per line")
    args = parser.parse_args()

    names = read_names(args.input)
    limiter = RateLimiter(DEFAULT_RATES)

    with make_client() as http:
        client = TrackedClient(http, limiter, 10)
        for name in names:
            exists = generals_api.user_exists(client, name)
            print(f"{'yes' if exists else 'no'}\t{name}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
