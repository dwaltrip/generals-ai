"""Build `quality.md` — the per-run training-quality report.

The quality counterpart to `bc.perf_report`: where that report asks "was the
GPU fed?", this one asks "did the heads learn?". A standing-panel summary
(best-epoch val value loss, the train−val value gap at that epoch, policy
top-1/3, pass accuracy next to its base rate) over the per-epoch metric
table. Built from `run_comparison`'s column definitions, so it stays in
lockstep with `compare_runs.py`.

Written at the end of every run by `bc.run_instrumentation`; regenerate
offline with `scripts/build_run_report.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

from training.analysis.run_comparison import (
    build_cols,
    build_summary_table,
    build_table,
    load_epochs,
)


# Per-epoch columns: the defaults plus the optional quality columns
# (gap, pass_*) — everything in the registry except perf (sps).
_EPOCH_COLS = [
    "t_pol", "t_val", "t_pass", "t_tot",
    "v_pol", "v_val", "v_pass", "v_tot",
    "gap", "top1", "top3", "pass_frac", "pass_acc",
]


def _read_args(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "args.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _render_header(args: dict) -> str | None:
    """One config line with the quality-relevant knobs: data, value-head
    variant, dense-history n, and the optimizer basics."""
    parts: list[str] = []
    if args.get("manifest"):
        parts.append(Path(args["manifest"]).name)
    arch = args.get("arch") or {}
    if arch.get("value_head_variant"):
        parts.append(f"value_head={arch['value_head_variant']}")
    n = (arch.get("obs") or {}).get("dense_history_n")
    if n is not None:
        parts.append(f"n={n}")
    if args.get("batch_size"):
        parts.append(f"bs={args['batch_size']}")
    if args.get("lr") is not None:
        parts.append(f"lr={args['lr']:g}")
    if args.get("weight_decay") is not None:
        parts.append(f"wd={args['weight_decay']:g}")
    return "**config** " + " · ".join(parts) if parts else None


def build_quality_report(run_dir: Path) -> str:
    """Render the full markdown quality report for one run dir."""
    parts = [f"# Quality report — {run_dir.name}"]
    header = _render_header(_read_args(run_dir))
    if header:
        parts.append(header)

    try:
        epochs = load_epochs(run_dir)
    except FileNotFoundError:
        epochs = []
    if not epochs:
        parts.append("*no epoch records — epochs.jsonl absent or empty*")
        return "\n\n".join(parts) + "\n"

    run = [(run_dir.name, epochs)]
    parts.append("## Summary\n\n" + build_summary_table(run))
    cols = [c for c in build_cols(dp=None) if c.name in _EPOCH_COLS]
    parts.append("## Per-epoch\n\n" + build_table(run, cols))
    return "\n\n".join(parts) + "\n"
