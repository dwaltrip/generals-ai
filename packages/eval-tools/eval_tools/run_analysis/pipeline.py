"""Drives the run-analysis stages end to end: extract rows from a finished
run dir (or reuse previously written CSVs), aggregate, and render the report
artifacts into the output dir.

`analyze_run` is the single entry point, shared by the analyze_eval.py CLI
and the eval runner (which invokes it at the end of each run).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from eval_tools.run_analysis import aggregate, extract, loader, report


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


def analyze_run(
    run_dir: Path,
    out_dir: Path | None = None,
    *,
    labels: list[str] | None = None,
    reuse: bool = False,
    plots: bool = True,
) -> Path:
    """Analyze a finished eval run; returns the path to the rendered report.md.

    `reuse` re-renders from existing CSVs without re-reading the game npz
    files. `labels` overrides the auto-derived group labels (one per distinct
    policy spec).
    """
    out_dir = out_dir or run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = loader.load_config(run_dir)
    groups = loader.derive_groups(config["policy_specs"], labels)
    group_labels = [g.label for g in groups]
    print(f"groups: {', '.join(f'{g.label} (slots {g.policy_indices})' for g in groups)}")

    pg_csv = out_dir / "player_games.csv"
    ss_csv = out_dir / "stage_snapshots.csv"
    if reuse and pg_csv.exists() and ss_csv.exists():
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
    report_path = out_dir / "report.md"
    report_path.write_text(
        report.render_report(agg, config, groups, run_dir.name)
    )
    (out_dir / "distributions.md").write_text(
        report.render_distributions(agg, group_labels)
    )
    if plots:
        for p in report.make_plots(stage_rows, agg, group_labels, out_dir):
            print(f"wrote {p}")
    print(f"report: {report_path}")
    return report_path
