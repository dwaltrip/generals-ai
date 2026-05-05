import argparse

from replay_collector.cli import collect_recent, sweep_metadata


def main() -> None:
    parser = argparse.ArgumentParser(prog="replay_collector")
    sub = parser.add_subparsers(dest="command", required=True)
    collect_recent.add_parser(sub)
    sweep_metadata.add_parser(sub)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
