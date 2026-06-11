#!/usr/bin/env -S uv run python
"""Analyze a finished eval run: extract per-game rows, aggregate, render.

Usage:
    ./scripts/analyze_eval.py data/eval-runs/<timestamp> [--labels A,B] [--reuse]

Writes to <run-dir>/analysis/:
    player_games.csv, stage_snapshots.csv   — flat row tables (the reusable seam)
    metrics.json                            — all aggregates
    report.md, distributions.md             — rendered tables
    *.png                                   — share curves

--reuse re-renders from existing CSVs without re-reading the game npz files.

The runner produces the same analysis automatically at the end of each eval
run; this CLI re-runs it post hoc (e.g. with custom labels, or after analysis
code changes).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_tools.run_analysis.pipeline import analyze_run
from utils.docstring import doc_summary


def main() -> None:
    ap = argparse.ArgumentParser(description=doc_summary(__doc__))
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <run-dir>/analysis")
    ap.add_argument("--labels", type=str, default=None,
                    help="comma-separated group labels, one per distinct policy spec")
    ap.add_argument("--reuse", action="store_true",
                    help="re-render from existing CSVs; skip npz extraction")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    analyze_run(
        args.run_dir,
        args.out,
        labels=args.labels.split(",") if args.labels else None,
        reuse=args.reuse,
        plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
