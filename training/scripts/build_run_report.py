#!/usr/bin/env -S uv run python
"""CLI wrapper around `bc.run_report.build_report`.

Point it at a pulled run dir; prints the markdown report to stdout, or writes
it to a file with `-o PATH`.

    training/scripts/build_run_report.py training/data/runs-cloud/<run>
    training/scripts/build_run_report.py <run> -o report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from bc.run_report import build_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path, help="path to a run directory")
    parser.add_argument(
        "-o", "--out", type=Path, default=None,
        help="write the report to this path (default: stdout)",
    )
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        parser.error(f"not a directory: {args.run_dir}")

    report = build_report(args.run_dir)
    if args.out is None:
        print(report)
    else:
        args.out.write_text(report)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
