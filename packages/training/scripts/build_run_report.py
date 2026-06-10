#!/usr/bin/env -S uv run python
"""Regenerate a run dir's markdown reports offline.

Point it at a run dir (e.g. a pulled cloud run); prints the quality report to
stdout by default. `--perf` prints the perf report instead; `--write` writes
both `quality.md` and `perf.md` into the run dir.

    build_run_report.py data/training/runs-cloud/<run>
    build_run_report.py <run> --perf
    build_run_report.py <run> --write
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from training.analysis.quality_report import build_quality_report
from training.bc.perf_report import build_perf_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path, help="path to a run directory")
    parser.add_argument(
        "--perf", action="store_true",
        help="print the perf report instead of the quality report",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="write quality.md and perf.md into the run dir instead of printing",
    )
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        parser.error(f"not a directory: {args.run_dir}")

    if args.write:
        for name, builder in [
            ("quality.md", build_quality_report),
            ("perf.md", build_perf_report),
        ]:
            path = args.run_dir / name
            path.write_text(builder(args.run_dir))
            print(f"wrote {path}", file=sys.stderr)
    else:
        builder = build_perf_report if args.perf else build_quality_report
        print(builder(args.run_dir))


if __name__ == "__main__":
    main()
