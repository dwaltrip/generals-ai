"""Markdown tables and per-run summaries over training-run epoch metrics.

Reads per-epoch records from a run dir's `epochs.jsonl`. Columns are defined
once in a registry, with numeric extraction (`ColDef.extract`) kept separate
from formatting (`ColDef.fmt`) so derived columns (the train−val value gap)
and cross-epoch reductions (`summarize_run`, the best-epoch summary) share
the same definitions as the plain per-epoch tables.

Consumers: `scripts/compare_runs.py` (CLI, multi-run) and
`training.analysis.quality_report` (per-run `quality.md`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path
from typing import NamedTuple

from utils.format import format_loss, format_pct, format_rate, md_table


MISSING = "—"

Extract = Callable[[dict], float | None]
Fmt = Callable[[float], str]


@dataclass(frozen=True)
class ColDef:
    """One table column. `extract` pulls the numeric value from an epoch
    record (None when absent); `fmt` renders it for display. Keeping the two
    separate is what lets derived columns and summary reductions reuse the
    same definitions instead of re-parsing records."""

    name: str
    extract: Extract
    fmt: Fmt
    long_name: str = ""
    align: str = "right"
    optional: bool = False

    @property
    def display_name(self) -> str:
        return self.long_name or self.name

    def cell(self, e: dict) -> str | None:
        v = self.extract(e)
        return self.fmt(v) if v is not None else None


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

class ColSpec(NamedTuple):
    name: str
    kind: str  # "loss" | "pct" | "rate" — picks the formatter in build_cols()
    optional: bool = False

# Authoritative ordering; formatters attached at runtime by build_cols()
# so they can capture --dp.
_COL_REGISTRY: list[ColSpec] = [
    ColSpec("t_pol", "loss"),
    ColSpec("t_val", "loss"),
    ColSpec("t_vsoft", "loss", optional=True),
    ColSpec("t_pass", "loss"),
    ColSpec("t_tot", "loss"),
    ColSpec("v_pol", "loss"),
    ColSpec("v_val", "loss"),
    ColSpec("v_vsoft", "loss", optional=True),
    ColSpec("v_pass", "loss"),
    ColSpec("v_tot", "loss"),
    ColSpec("gap", "loss", optional=True),
    ColSpec("top1", "pct"),
    ColSpec("top3", "pct"),
    ColSpec("v_H", "loss", optional=True),
    ColSpec("v_eH", "eff", optional=True),
    ColSpec("sps", "rate", optional=True),
    ColSpec("pass_frac", "pct", optional=True),
    ColSpec("pass_acc", "pct", optional=True),
    # Elim head (optional; present only on elim runs). t_/v_ are train/val CE;
    # e_top1 is top-1 bin accuracy, e_H the prediction entropy (collapse check).
    ColSpec("t_elim", "loss", optional=True),
    ColSpec("t_esoft", "loss", optional=True),
    ColSpec("v_elim", "loss", optional=True),
    ColSpec("v_esoft", "loss", optional=True),
    ColSpec("e_top1", "pct", optional=True),
    ColSpec("e_H", "loss", optional=True),
]

ALL_COL_NAMES = [col.name for col in _COL_REGISTRY]

LONG_NAMES: dict[str, str] = {
    "t_pol": "train_policy",
    "t_val": "train_value",
    "t_vsoft": "train_value_soft",
    "t_pass": "train_pass",
    "t_tot": "train_total",
    "v_pol": "val_policy",
    "v_val": "val_value",
    "v_vsoft": "val_value_soft",
    "v_pass": "val_pass",
    "v_tot": "val_total",
    "gap": "value_gap",
    "t_elim": "train_elim",
    "t_esoft": "train_elim_soft",
    "v_elim": "val_elim",
    "v_esoft": "val_elim_soft",
    "e_top1": "val_elim_top1",
    "e_H": "val_elim_entropy",
}

# The train/val/top groups cover non-optional columns only; optional ones are
# otherwise summoned by name. `elim` is the exception — a named group for the
# (all-optional) elim-head columns, so an elim run's metrics come in via one
# token (`--cols elim`).
_DEFAULT_NAMES = [col.name for col in _COL_REGISTRY if not col.optional]

_ELIM_NAMES = ["t_elim", "t_esoft", "v_elim", "v_esoft", "e_top1", "e_H"]

GROUPS: dict[str, list[str]] = {
    "train": [n for n in _DEFAULT_NAMES if n.startswith("t_")],
    "val": [n for n in _DEFAULT_NAMES if n.startswith("v_") or n.startswith("top")],
    "top": [n for n in _DEFAULT_NAMES if n.startswith("top")],
    "elim": _ELIM_NAMES,
}

VALID_TOKENS = sorted(set(ALL_COL_NAMES) | set(GROUPS))

_COL_REFERENCE = """\

Available columns:
  train:  t_pol  t_val  t_pass  t_tot
  val:    v_pol  v_val  v_pass  v_tot  top1  top3
  optional (hidden by default):  gap  t_vsoft  v_vsoft  v_H  v_eH  sps  pass_frac  pass_acc
  elim (optional; --cols elim):  t_elim  t_esoft  v_elim  v_esoft  e_top1  e_H

Groups (expand to all in category):  train  val  top  elim

Naming: t_* is short for train_*, v_* for val_*; gap is v_val − t_val;
v_H is mean val policy entropy (nats, non-pass frames), v_eH = e^(v_H).
elim: t_/v_elim are train/val elim CE; e_top1 is top-1 bin accuracy,
e_H the elim prediction entropy (nats)."""


def _val(e: dict) -> dict:
    return e.get("val") or {}


def _first(*vals: float | None) -> float | None:
    """First non-None value — coalesce a metric across renamed keys
    (e.g. older runs log `elim`, newer ones `next_elim`)."""
    return next((v for v in vals if v is not None), None)


def _value_gap(e: dict) -> float | None:
    """val_value − train_value. Positive and growing across epochs is the
    value-head memorization signature (train falls while val doesn't follow)."""
    t, v = e.get("value"), _val(e).get("value")
    return v - t if t is not None and v is not None else None


def _policy_perplexity(e: dict) -> float | None:
    """e^(mean policy entropy) — the "effective number of actions" the
    policy is choosing among. Derived from the same jsonl field as `v_H`."""
    h = _val(e).get("policy_entropy")
    return math.exp(h) if h is not None else None


_EXTRACTORS: dict[str, Extract] = {
    "t_pol":      lambda e: e.get("policy"),
    "t_val":      lambda e: e.get("value"),
    "t_vsoft":    lambda e: e.get("value_soft"),
    "t_pass":     lambda e: e.get("pass"),
    "t_tot":      lambda e: e.get("total"),
    "v_pol":      lambda e: _val(e).get("policy"),
    "v_val":      lambda e: _val(e).get("value"),
    "v_vsoft":    lambda e: _val(e).get("value_soft"),
    "v_pass":     lambda e: _val(e).get("pass"),
    "v_tot":      lambda e: _val(e).get("total"),
    "gap":        _value_gap,
    "top1":       lambda e: _val(e).get("top1"),
    "top3":       lambda e: _val(e).get("top3"),
    "v_H":        lambda e: _val(e).get("policy_entropy"),
    "v_eH":       _policy_perplexity,
    "sps":        lambda e: e.get("samples_per_sec"),
    "pass_frac":  lambda e: _val(e).get("pass_frac"),
    "pass_acc":   lambda e: _val(e).get("pass_acc"),
    "t_elim":     lambda e: _first(e.get("elim"), e.get("next_elim")),
    "t_esoft":    lambda e: _first(e.get("elim_soft"), e.get("next_elim_soft")),
    "v_elim":     lambda e: _first(_val(e).get("elim"), _val(e).get("next_elim")),
    "v_esoft":    lambda e: _val(e).get("elim_soft"),
    "e_top1":     lambda e: _val(e).get("elim_top1"),
    "e_H":        lambda e: _val(e).get("elim_pred_entropy"),
}


def build_cols(dp: int | None, short_names: bool = False) -> list[ColDef]:
    fmts: dict[str, Fmt] = {
        "loss": lambda v: format_loss(v, dp=dp),
        "pct": format_pct,
        "rate": format_rate,
        # Effective counts (e.g. v_eH, ~1–50): one decimal place.
        "eff": lambda v: f"{v:.1f}",
    }

    registry_names = {col.name for col in _COL_REGISTRY}
    assert registry_names == set(_EXTRACTORS), (
        f"_COL_REGISTRY / _EXTRACTORS mismatch: "
        f"missing extractors {registry_names - set(_EXTRACTORS)}, "
        f"extra extractors {set(_EXTRACTORS) - registry_names}"
    )

    return [
        ColDef(
            name=col.name,
            extract=_EXTRACTORS[col.name],
            fmt=fmts[col.kind],
            long_name="" if short_names else LONG_NAMES.get(col.name, ""),
            optional=col.optional,
        )
        for col in _COL_REGISTRY
    ]


# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------

def _expand_names(raw: str) -> list[str]:
    """Expand comma-separated column names / group shorthands.

    Raises ValueError on unknown tokens.
    """
    names: list[str] = []
    col_set = set(ALL_COL_NAMES)
    for token in raw.split(","):
        token = token.strip()
        if token in GROUPS:
            names.extend(GROUPS[token])
        elif token in col_set:
            names.append(token)
        else:
            raise ValueError(
                f"unknown column or group '{token}'\n{_COL_REFERENCE}"
            )
    return names


def resolve_cols(
    all_cols: list[ColDef],
    cols_arg: str | None,
    exclude_arg: str | None,
) -> list[ColDef]:
    """Raises ValueError on bad input."""
    if cols_arg is not None and exclude_arg is not None:
        raise ValueError("--cols and --exclude are mutually exclusive")

    col_map = {col.name: col for col in all_cols}

    if cols_arg is not None:
        seen: set[str] = set()
        result: list[ColDef] = []
        for name in _expand_names(cols_arg):
            if name not in seen:
                seen.add(name)
                result.append(col_map[name])
        return result

    defaults = [col for col in all_cols if not col.optional]

    if exclude_arg is not None:
        drop = set(_expand_names(exclude_arg))
        return [col for col in defaults if col.name not in drop]

    return defaults


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------

def parse_run_arg(raw: str, base: Path | None) -> tuple[Path, str | None]:
    if "," in raw:
        path_str, label = raw.rsplit(",", 1)
    else:
        path_str, label = raw, None
    p = Path(path_str)
    if base is not None and not p.is_absolute() and not p.exists():
        p = base / path_str
    return p, label


# ---------------------------------------------------------------------------
# Per-run summary
# ---------------------------------------------------------------------------

def summarize_run(epochs: list[dict]) -> dict[str, float | int | None]:
    """Reduce a run's epoch records to its standing-panel scalars.

    `best_*` keys are anchored at the epoch with minimum `v_val` — the value
    head's best epoch typically lands earlier than the policy's, so the final
    epoch under-reports it. `last_*` keys read the final epoch. Any value is
    None when its source field is absent.
    """
    ex = _EXTRACTORS
    last = epochs[-1] if epochs else {}
    candidates = [
        (i, v) for i, e in enumerate(epochs) if (v := ex["v_val"](e)) is not None
    ]
    best_i, best_v = min(candidates, key=lambda iv: iv[1]) if candidates else (None, None)
    return {
        "n_epochs": len(epochs),
        "best_v_val": best_v,
        "best_epoch": best_i + 1 if best_i is not None else None,
        "gap_at_best": ex["gap"](epochs[best_i]) if best_i is not None else None,
        "last_v_val": ex["v_val"](last),
        "last_v_pol": ex["v_pol"](last),
        "last_top1": ex["top1"](last),
        "last_top3": ex["top3"](last),
        "last_pass_acc": ex["pass_acc"](last),
        "last_pass_frac": ex["pass_frac"](last),
    }


_SUMMARY_NOTE = (
    "*best v_val = the run's minimum; gap@best = v_val − t_val at that epoch "
    "(large positive = memorization); all other columns are final-epoch.*"
)

_FLOOR_NOTE = (
    "*floor = the manifest's frame-weighted val marginal entropy; "
    "Δbest = best v_val − floor (negative = beat the marginal predictor).*"
)


def build_summary_table(
    runs: list[tuple[str, list[dict]]],
    dp: int | None = None,
    floors: list[float | None] | None = None,
    notes: bool = True,
) -> str:
    """One row per run: best-epoch val value loss (+ the value gap there) and
    final-epoch quality metrics. The cross-run counterpart of the per-epoch
    tables; also the standing-panel summary in `quality.md`.

    `floors` holds each run's marginal-entropy floor: the `val_value` a model
    would score by ignoring the board and predicting the val split's placement
    frequencies (so it's a property of the run's manifest, not of the model).
    It anchors the table's value-loss columns — below floor means the head is
    extracting signal from the input, at floor honest-but-uninformative, above
    floor confidently wrong. One entry per run, aligned with `runs` by index,
    None where unknown; callers resolve the values via
    `run_metrics.floor_for_run` (an IO concern, deliberately kept out of this
    module). When `floors` is None entirely, the floor/Δbest columns are
    omitted.
    """

    def loss(v: float | None) -> str:
        return format_loss(v, dp=dp) if v is not None else MISSING

    def pct(v: float | None) -> str:
        return format_pct(v) if v is not None else MISSING

    single = len(runs) == 1
    headers = ([] if single else ["run"]) + [
        "epochs", "best v_val", "@ep",
        *(["floor", "Δbest"] if floors is not None else []),
        "last v_val", "gap@best",
        "v_pol", "top1", "top3", "pass_acc", "pass_frac",
    ]
    rows: list[list[object]] = []
    for ri, (label, epochs) in enumerate(runs):
        s = summarize_run(epochs)
        floor = floors[ri] if floors is not None else None
        best_v = s["best_v_val"]
        row: list[object] = [] if single else [label]
        row += [
            s["n_epochs"],
            loss(best_v),
            s["best_epoch"] if s["best_epoch"] is not None else MISSING,
        ]
        if floors is not None:
            delta = best_v - floor if best_v is not None and floor is not None else None
            row += [loss(floor), loss(delta)]
        row += [
            loss(s["last_v_val"]),
            loss(s["gap_at_best"]),
            loss(s["last_v_pol"]),
            pct(s["last_top1"]),
            pct(s["last_top3"]),
            pct(s["last_pass_acc"]),
            pct(s["last_pass_frac"]),
        ]
        rows.append(row)
    aligns = ([] if single else ["left"]) + ["right"] * (len(headers) - (0 if single else 1))
    table = md_table(headers, rows, align=aligns)
    if not notes:
        return table
    note = _SUMMARY_NOTE + ("\n" + _FLOOR_NOTE if floors is not None else "")
    return table + "\n\n" + note


# ---------------------------------------------------------------------------
# Extraction + rendering
# ---------------------------------------------------------------------------

def _extract(
    runs: list[tuple[str, list[dict]]],
    cols: list[ColDef],
) -> tuple[int, list[list[list[str]]]]:
    """Returns (max_epochs, grid) where grid[epoch_idx][col_idx][run_idx] is a formatted cell."""
    max_epochs = max(len(epochs) for _, epochs in runs)
    grid: list[list[list[str]]] = []
    for ei in range(max_epochs):
        epoch_cols: list[list[str]] = []
        for col in cols:
            run_cells: list[str] = []
            for _, epochs in runs:
                if ei >= len(epochs):
                    run_cells.append(MISSING)
                else:
                    val = col.cell(epochs[ei])
                    run_cells.append(val if val is not None else MISSING)
            epoch_cols.append(run_cells)
        grid.append(epoch_cols)
    return max_epochs, grid


def build_table(runs: list[tuple[str, list[dict]]], cols: list[ColDef]) -> str:
    # A single run needs no "run" column to disambiguate rows — drop it.
    single = len(runs) == 1
    headers = ["epoch"] + ([] if single else ["run"]) + [col.display_name for col in cols]
    aligns = ["right"] + ([] if single else ["left"]) + [col.align for col in cols]

    max_epochs, grid = _extract(runs, cols)
    labels = [label for label, _ in runs]
    rows = []
    for ei in range(max_epochs):
        for ri, label in enumerate(labels):
            cells = [grid[ei][ci][ri] for ci in range(len(cols))]
            prefix = [ei + 1] if single else [ei + 1, label]
            rows.append(prefix + cells)

    return md_table(headers, rows, align=aligns)


def build_wide_table(runs: list[tuple[str, list[dict]]], cols: list[ColDef]) -> str:
    labels = [label for label, _ in runs]
    n_runs = len(runs)

    max_epochs, grid = _extract(runs, cols)

    # Physical columns: [epoch] + [col0_run0, col0_run1, ...] + [col1_run0, ...]
    # Metric names go in the header row; run labels become the first data row.
    headers = ["epoch"]
    for col in cols:
        headers.append(col.name)
        headers.extend([""] * (n_runs - 1))

    label_row: list[object] = [""]
    for _ in cols:
        label_row.extend(labels)

    data_rows: list[list[object]] = []
    for ei in range(max_epochs):
        row: list[object] = [ei + 1]
        for ci in range(len(cols)):
            row.extend(grid[ei][ci])
        data_rows.append(row)

    aligns = ["right"] * len(headers)
    return md_table(headers, [label_row] + data_rows, align=aligns)
