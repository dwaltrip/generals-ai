#!/usr/bin/env -S uv run python
"""Compare epoch metrics across training runs.

Usage examples:

    compare_runs.py run1[,label1] run2[,label2] ...
    compare_runs.py --base data/training/runs ts1 ts2
    compare_runs.py --all --base data/training/sweeps/my-sweep/runs
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

`--all` discovers every dir under `--base` that contains `epochs.jsonl`.
When a symlink and its target both exist, the symlink is kept (sweep tooling
gives symlinks meaningful names). Capped at 15 runs; list explicitly for more.

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
from datetime import UTC, datetime
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


_MAX_ALL_RUNS = 15

_TIMESTAMP_FMTS = ("%Y-%m-%dT%H-%M-%SZ", "%Y%m%d-%H%M%S")


def _parse_run_timestamp(name: str) -> datetime | None:
    for fmt in _TIMESTAMP_FMTS:
        try:
            return datetime.strptime(name, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _label_runs(pairs: list[tuple[Path, str | None]]) -> list[tuple[Path, str]]:
    """Fill in labels where not user-provided.

    Timestamp dir names get compact date formatting; other names pass through.
    Year is included only when auto-labeled entries span multiple years.
    """
    parsed = [(p, given, _parse_run_timestamp(p.name)) for p, given in pairs]
    auto_years = {dt.year for _, given, dt in parsed if given is None and dt is not None}
    include_year = len(auto_years) > 1

    results: list[tuple[Path, str]] = []
    for p, given, dt in parsed:
        if given is not None:
            label = given
        elif dt is None:
            label = p.name
        elif include_year:
            label = dt.strftime("%Y-%-m-%d %H:%M:%S")
        else:
            label = dt.strftime("%-m-%d %H:%M:%S")
        results.append((p, label))
    return results


def discover_runs(base: Path) -> list[tuple[Path, str]]:
    """All epoch-bearing dirs under *base*, symlinks preferred over targets.

    Returns (run_dir, label) pairs sorted by directory name.
    """
    by_real: dict[Path, list[tuple[Path, bool]]] = {}
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        if not (entry / "epochs.jsonl").is_file():
            continue
        real = entry.resolve()
        by_real.setdefault(real, []).append((entry, entry.is_symlink()))

    chosen: list[Path] = []
    for entries in by_real.values():
        symlinks = sorted((p for p, is_sym in entries if is_sym), key=lambda p: p.name)
        if symlinks:
            chosen.append(symlinks[0])
        else:
            chosen.append(entries[0][0])

    chosen.sort(key=lambda p: p.name)
    return _label_runs([(p, None) for p in chosen])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "runs", nargs="*", metavar="RUN_DIR[,LABEL]",
        help="run directory, optionally with a display label after a comma",
    )
    parser.add_argument(
        "--all", "-a", action="store_true",
        help="compare all runs under --base that contain epochs.jsonl "
             f"(max {_MAX_ALL_RUNS}; symlinks preferred over their targets)",
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
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="show explanatory footnotes below tables",
    )
    return parser


def _resolve_run_pairs(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> list[tuple[Path, str]]:
    if args.all and args.runs:
        parser.error("--all and positional run arguments are mutually exclusive")
    if args.all and args.base is None:
        parser.error("--all requires --base")
    if not args.all and not args.runs:
        parser.error("provide run arguments, or use --all with --base")

    if args.all:
        pairs = discover_runs(args.base)
        if not pairs:
            parser.error(f"no runs with epochs.jsonl found in {args.base}")
        if len(pairs) > _MAX_ALL_RUNS:
            parser.error(
                f"--all found {len(pairs)} runs in {args.base}, "
                f"max is {_MAX_ALL_RUNS} — list runs explicitly to compare more"
            )
        return pairs

    raw_pairs: list[tuple[Path, str | None]] = []
    for raw in args.runs:
        run_dir, label = parse_run_arg(raw, args.base)
        if not run_dir.is_dir():
            parser.error(f"not a directory: {run_dir}")
        raw_pairs.append((run_dir, label))
    return _label_runs(raw_pairs)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    all_cols = build_cols(args.dp, short_names=args.short_names)
    try:
        cols = resolve_cols(all_cols, args.cols, args.exclude)
    except ValueError as exc:
        parser.error(str(exc))

    pairs = _resolve_run_pairs(args, parser)
    runs: list[tuple[str, list[dict]]] = []
    run_dirs: list[Path] = []
    for run_dir, label in pairs:
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
        parts.append(build_summary_table(
            runs, dp=args.dp, floors=floors, notes=args.verbose,
        ))
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
