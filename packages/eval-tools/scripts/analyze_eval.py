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
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from eval_tools.run_analysis import aggregate, extract, loader, report
from utils.docstring import doc_summary


_STRING_COLS = {"game_id", "replay_id", "group", "axis"}
_INT_COLS = {
    "game_index", "seat", "policy_idx", "won", "is_draw_game", "game_length",
    "death_tick", "killed_by_seat", "kills", "stage", "span_start", "span_end",
    "t_mid", "eventual_win", "total_moves", "total_passes", "ones_traversed",
    "n_no_legal",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            row = {}
            for k, v in raw.items():
                if k in _STRING_COLS:
                    row[k] = v
                elif k in _INT_COLS:
                    row[k] = int(float(v)) if v not in ("", "nan") else -1
                else:
                    row[k] = float(v) if v else math.nan
            rows.append(row)
    return rows


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

    run_dir: Path = args.run_dir
    out_dir: Path = args.out or run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = loader.load_config(run_dir)
    labels = args.labels.split(",") if args.labels else None
    groups = loader.derive_groups(config["policy_specs"], labels)
    group_labels = [g.label for g in groups]
    print(f"groups: {', '.join(f'{g.label} (slots {g.policy_indices})' for g in groups)}")

    pg_csv = out_dir / "player_games.csv"
    ss_csv = out_dir / "stage_snapshots.csv"
    if args.reuse and pg_csv.exists() and ss_csv.exists():
        player_rows, stage_rows = read_csv(pg_csv), read_csv(ss_csv)
        print(f"reused {len(player_rows)} player rows, {len(stage_rows)} stage rows")
    else:
        player_rows, stage_rows = [], []
        n = 0
        for rec in loader.iter_games(run_dir):
            prs, srs = extract.extract_game(rec, groups)
            player_rows += prs
            stage_rows += srs
            n += 1
        write_csv(pg_csv, player_rows)
        write_csv(ss_csv, stage_rows)
        print(f"extracted {n} games → {len(player_rows)} player rows, {len(stage_rows)} stage rows")

    agg = aggregate.aggregate_all(player_rows, stage_rows, group_labels)
    (out_dir / "metrics.json").write_text(json.dumps(agg, indent=1))
    (out_dir / "report.md").write_text(
        report.render_report(agg, config, groups, run_dir.name)
    )
    (out_dir / "distributions.md").write_text(
        report.render_distributions(agg, group_labels)
    )
    if not args.no_plots:
        for p in report.make_plots(stage_rows, agg, group_labels, out_dir):
            print(f"wrote {p}")
    print(f"report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
