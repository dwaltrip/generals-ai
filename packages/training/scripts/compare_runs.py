#!/usr/bin/env -S uv run python
"""Compare epoch metrics across training runs.

Usage examples:

    compare_runs.py run1[,label1] run2[,label2] ...
    compare_runs.py --base data/training/runs ts1 ts2
    compare_runs.py --wide run1[,label1] run2[,label2] ...
    compare_runs.py --cols t_pol,v_pol,top1 run1 run2
    compare_runs.py --cols val,sps run1 run2
    compare_runs.py --exclude train run1 run2
    compare_runs.py --dp 5 run1 run2

Each positional arg is a run directory (or a leaf name resolved under --base),
optionally followed by a comma and a display label. Prints a markdown table
to stdout with train and val metrics per epoch, reading from `epochs.jsonl`.

Column reference for --cols / --exclude:

  train:  t_pol  t_val  t_pass  t_tot
  val:    v_pol  v_val  v_pass  v_tot  top1  top3
  optional (hidden by default):  sps  pass_frac  pass_acc

  Groups (expand to all in category):  train  val  top
  Naming: t_* is short for train_*, v_* for val_*

Pivot the table with `--wide`. This groups values per run as sub-columns under
each metric. Each epoch becomes a single row, instead of a row per (epoch, run).

This is a CLI-wrapper around `run_comparison.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from training.analysis.run_comparison import (
    build_cols,
    build_table,
    build_wide_table,
    load_epochs,
    parse_run_arg,
    resolve_cols,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "runs", nargs="+", metavar="RUN_DIR[,LABEL]",
        help="run directory, optionally with a display label after a comma",
    )
    parser.add_argument(
        "--base", "-b", type=Path, default=None,
        help="base directory — positional args that aren't valid paths are "
             "resolved relative to this",
    )
    parser.add_argument(
        "--wide", "-w", action="store_true",
        help="pivot table: one row per epoch, metrics as top-level columns "
             "with one sub-column per run",
    )
    parser.add_argument(
        "--cols", "-c", default=None, metavar="COL,COL,...",
        help="e.g. --cols t_pol,top1,sps or --cols val,top "
             "(see column reference above for all names and groups)",
    )
    parser.add_argument(
        "--exclude", "-x", default=None, metavar="COL,COL,...",
        help="e.g. --exclude train or --exclude t_val,top3 "
             "(same syntax as --cols)",
    )
    parser.add_argument(
        "--dp", type=int, default=None, metavar="N",
        help="decimal places for loss columns (default: adaptive — 4 dp "
             "below 1, 3 dp above)",
    )
    args = parser.parse_args()

    all_cols = build_cols(args.dp)
    try:
        cols = resolve_cols(all_cols, args.cols, args.exclude)
    except ValueError as exc:
        parser.error(str(exc))

    runs: list[tuple[str, list[dict]]] = []
    for raw in args.runs:
        run_dir, label = parse_run_arg(raw, args.base)
        if not run_dir.is_dir():
            parser.error(f"not a directory: {run_dir}")
        epochs = load_epochs(run_dir)
        runs.append((label, epochs))

    max_epochs = max(len(epochs) for _, epochs in runs)
    if max_epochs == 0:
        print("no epochs found", file=sys.stderr)
        sys.exit(1)

    if args.wide:
        print(build_wide_table(runs, cols))
    else:
        print(build_table(runs, cols))


if __name__ == "__main__":
    main()
