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
    compare_runs.py --summary-only run1 run2 run3

Each positional arg is a run directory (or a leaf name resolved under --base),
optionally followed by a comma and a display label. Prints a markdown table
to stdout with train and val metrics per epoch, reading from `epochs.jsonl`.
With multiple runs, a per-run summary table (best-epoch val_value, the value
gap at that epoch, final-epoch quality) precedes the per-epoch table;
`--summary-only` skips the per-epoch table, `--no-summary` the summary. The
summary includes each run's manifest floor (resolved via the manifest's
metrics sidecar, backfilled on first use) unless `--no-floors`.

Column reference for --cols / --exclude:

  train:  t_pol  t_val  t_pass  t_tot
  val:    v_pol  v_val  v_pass  v_tot  top1  top3
  optional (hidden by default):  gap  t_vsoft  v_vsoft  sps  pass_frac  pass_acc

  Groups (expand to all in category):  train  val  top
  Naming: t_* is short for train_*, v_* for val_*; gap is v_val − t_val

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
    build_summary_table,
    build_table,
    build_wide_table,
    load_epochs,
    parse_run_arg,
    resolve_cols,
)
from training.analysis.run_metrics import floor_for_run


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
        "--short-names", action="store_true",
        help="use the short column names (t_pol, v_val, …) as table headers",
    )
    parser.add_argument(
        "--dp", type=int, default=None, metavar="N",
        help="decimal places for loss columns (default: adaptive — 4 dp "
             "below 1, 3 dp above)",
    )
    summary_group = parser.add_mutually_exclusive_group()
    summary_group.add_argument(
        "--summary-only", "-s", action="store_true",
        help="print only the per-run summary table",
    )
    summary_group.add_argument(
        "--no-summary", action="store_true",
        help="print only the per-epoch table (default prepends the summary "
             "when comparing multiple runs)",
    )
    parser.add_argument(
        "--no-floors", action="store_true",
        help="skip the summary's per-manifest floor columns (default resolves "
             "each run's manifest and backfills its metrics sidecar if needed)",
    )
    args = parser.parse_args()

    all_cols = build_cols(args.dp, short_names=args.short_names)
    try:
        cols = resolve_cols(all_cols, args.cols, args.exclude)
    except ValueError as exc:
        parser.error(str(exc))

    runs: list[tuple[str, list[dict]]] = []
    run_dirs: list[Path] = []
    for i, raw in enumerate(args.runs):
        run_dir, label = parse_run_arg(raw, args.base)
        if not label:
            label = f"run-{i+1}"
        if not run_dir.is_dir():
            parser.error(f"not a directory: {run_dir}")
        try:
            epochs = load_epochs(run_dir)
        except FileNotFoundError:
            parser.error(f"no epochs.jsonl in {run_dir}")
        runs.append((label, epochs))
        run_dirs.append(run_dir)

    max_epochs = max(len(epochs) for _, epochs in runs)
    if max_epochs == 0:
        print("no epochs found", file=sys.stderr)
        sys.exit(1)

    parts: list[str] = []
    if args.summary_only or (len(runs) > 1 and not args.no_summary):
        floors = None
        if not args.no_floors:
            floors = [floor_for_run(d, create=True) for d in run_dirs]
        parts.append(build_summary_table(runs, dp=args.dp, floors=floors))
    if not args.summary_only:
        # --wide only matters if there are multiple runs to pivot into
        # sub-columns; for a single run, ignore the flag so it collapses to
        # the normal table.
        if args.wide and len(runs) > 1:
            parts.append(build_wide_table(runs, cols))
        else:
            parts.append(build_table(runs, cols))
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
