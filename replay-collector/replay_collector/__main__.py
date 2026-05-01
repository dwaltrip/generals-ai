import argparse
import logging

from replay_collector.collector import collect


def main() -> None:
    parser = argparse.ArgumentParser(prog="replay_collector")
    parser.add_argument("username", help="generals.io username (case-sensitive, spaces allowed)")
    parser.add_argument("--limit", type=int, default=5, help="max replays to fetch (default: 5)")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    collect(args.username, args.limit)


if __name__ == "__main__":
    main()
