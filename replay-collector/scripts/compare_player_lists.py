"""Compare a primary player-list txt file against one or more others.

Reports players from the primary file that either appear in (overlap) or
are absent from (unique) the union of the other files. Primary order is
preserved so meaningful orderings (e.g. leaderboard rank) survive.

Usage:
    uv run python scripts/compare_player_lists.py \\
        player_list_1.txt player_list_2.txt [player_list_3.txt ...] \\
        --overlap | --unique | --both
"""

import argparse
from pathlib import Path
import sys

from replay_collector.cli._shared import load_players_raw
from utils.docstring import doc_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=doc_summary(__doc__))
    parser.add_argument("primary", type=Path, help="primary player-list file")
    parser.add_argument("others", nargs="+", type=Path, help="one or more other player-list files")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overlap", action="store_true", help="primary players that appear in any other")
    mode.add_argument("--unique", action="store_true", help="primary players that appear in no other")
    mode.add_argument("--both", action="store_true", help="print both sections (default)")
    args = parser.parse_args()

    primary = load_players_raw(args.primary)
    others_union: set[str] = set()
    for path in args.others:
        others_union.update(load_players_raw(path))

    overlap = [n for n in primary if n in others_union]
    unique = [n for n in primary if n not in others_union]

    def emit_section(label: str, names: list[str]) -> None:
        print(f"# {label} ({len(names)})")
        for name in names:
            print(name)

    if args.overlap:
        for name in overlap:
            print(name)
    elif args.unique:
        for name in unique:
            print(name)
    else:
        emit_section("overlap", overlap)
        print()
        emit_section("unique", unique)

    print(
        f"primary={len(primary)} others_union={len(others_union)} "
        f"overlap={len(overlap)} unique={len(unique)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
