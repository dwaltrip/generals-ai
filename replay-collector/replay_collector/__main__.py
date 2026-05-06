import argparse
import sys

from replay_collector.cli import collect_recent, fetch_gior, sweep_metadata


def main() -> None:
    parser = argparse.ArgumentParser(prog="replay_collector")
    sub = parser.add_subparsers(dest="command", required=True)
    collect_recent.add_parser(sub)
    sweep_metadata.add_parser(sub)
    fetch_gior.add_parser(sub)
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        # 128 + SIGINT(2). Long-running runners catch this internally and
        # log a partial-progress summary; this is the safety net for any
        # interrupt outside those loops.
        sys.exit(130)


if __name__ == "__main__":
    main()
