"""Merge the top-N players from one or more leaderboard JSON files.

Each input is the JSON shape served by the generals.io leaderboard websocket:
a one-element list whose entry has `users` ranked best-first. We take the first
N from each file and dedupe across files, preserving first appearance — so a
player ranked highly in the first file beats their ranking in later files.

Usage:
    uv run python scripts/build_player_list.py \\
        data/2026-04-30-ffa-leaderboard.json \\
        data/2026-04-30-ffa-win-leaderboard.json \\
        --top 20 --output data/players.txt
"""

import argparse
import json
from pathlib import Path
import sys

from replay_collector.usernames import filter_valid
from utils.docstring import doc_summary


def load_users_from_json(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return filter_valid(data[0]["users"])


def main() -> None:
    parser = argparse.ArgumentParser(description=doc_summary(__doc__))
    parser.add_argument("inputs", nargs="+", type=Path, help="leaderboard JSON file(s)")
    parser.add_argument("--top", type=int, required=True, help="top-N per file")
    parser.add_argument("--output", type=Path, help="output path (default: stdout)")
    args = parser.parse_args()

    seen: set[str] = set()
    merged: list[str] = []
    for path in args.inputs:
        for name in load_users_from_json(path)[: args.top]:
            if name not in seen:
                seen.add(name)
                merged.append(name)

    text = "\n".join(merged) + "\n"
    if args.output:
        args.output.write_text(text)
        print(f"wrote {len(merged)} players to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
