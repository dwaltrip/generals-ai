"""Check which player names in a txt file exist on generals.io.

Reads one username per line and prints `<state>\\t<name>` to stdout where
<state> is one of:
    yes      — exists on generals.io
    no       — does not exist
    invalid  — fails our validity gate (layout-breaking or non-NFC); the
               name is printed via repr() so embedded control chars don't
               corrupt the output

Invalid names are reported, not skipped — this script is an audit tool, not
an action site. Uses `load_players_raw` so we see every line; the filtered
`load_players` would drop invalid names before we could label them.

Usage:
    uv run python scripts/check_users_exist.py data/players.txt
"""

import argparse
import csv
from pathlib import Path
import sys

from replay_collector import generals_api
from replay_collector.cli._shared import load_players_raw
from replay_collector.client import RateLimiter, TrackedClient, make_client
from replay_collector.runner import DEFAULT_RATES
from replay_collector.usernames import is_valid_username
from utils.docstring import doc_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=doc_summary(__doc__))
    parser.add_argument("input", type=Path, help="txt file with one username per line")
    args = parser.parse_args()

    names = load_players_raw(args.input)
    limiter = RateLimiter(DEFAULT_RATES)
    # excel-tab handles quoting if a name happens to contain "; default
    # lineterminator is \r\n which is awkward at a Unix terminal — pin \n.
    writer = csv.writer(sys.stdout, dialect="excel-tab", lineterminator="\n")

    with make_client() as http:
        client = TrackedClient(http, limiter, 10)
        for name in names:
            if not is_valid_username(name):
                writer.writerow(["invalid", repr(name)])
                sys.stdout.flush()
                continue
            exists = generals_api.user_exists(client, name)
            writer.writerow(["yes" if exists else "no", name])
            sys.stdout.flush()


if __name__ == "__main__":
    main()
