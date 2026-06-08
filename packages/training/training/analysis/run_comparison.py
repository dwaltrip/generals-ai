"""Build markdown tables for comparing training runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import NamedTuple

from utils.format import format_loss, format_pct, format_rate, md_table


MISSING = "—"

Extractor = Callable[[dict], str | None]


@dataclass(frozen=True)
class ColDef:
    name: str
    extractor: Extractor
    long_name: str = ""
    align: str = "right"
    optional: bool = False

    @property
    def display_name(self) -> str:
        return self.long_name or self.name


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

class ColSpec(NamedTuple):
    name: str
    optional: bool = False

# Authoritative ordering; extractors attached at runtime by _build_cols()
# so they can capture --dp.
_COL_REGISTRY: list[ColSpec] = [
    ColSpec("t_pol"),
    ColSpec("t_val"),
    ColSpec("t_pass"),
    ColSpec("t_tot"),
    ColSpec("v_pol"),
    ColSpec("v_val"),
    ColSpec("v_pass"),
    ColSpec("v_tot"),
    ColSpec("top1"),
    ColSpec("top3"),
    ColSpec("sps", optional=True),
    ColSpec("pass_frac", optional=True),
    ColSpec("pass_acc", optional=True),
]

ALL_COL_NAMES = [col.name for col in _COL_REGISTRY]

LONG_NAMES: dict[str, str] = {
    "t_pol": "train_policy",
    "t_val": "train_value",
    "t_pass": "train_pass",
    "t_tot": "train_total",
    "v_pol": "val_policy",
    "v_val": "val_value",
    "v_pass": "val_pass",
    "v_tot": "val_total",
}

GROUPS: dict[str, list[str]] = {
    "train": [n for n in ALL_COL_NAMES if n.startswith("t_")],
    "val": [n for n in ALL_COL_NAMES if n.startswith("v_") or n.startswith("top")],
    "top": [n for n in ALL_COL_NAMES if n.startswith("top")],
}

VALID_TOKENS = sorted(set(ALL_COL_NAMES) | set(GROUPS))

_COL_REFERENCE = """\

Available columns:
  train:  t_pol  t_val  t_pass  t_tot
  val:    v_pol  v_val  v_pass  v_tot  top1  top3
  optional (hidden by default):  sps  pass_frac  pass_acc

Groups (expand to all in category):  train  val  top

Naming: t_* is short for train_*, v_* for val_*"""


def _build_cols(dp: int | None) -> list[ColDef]:
    def loss(d: dict, key: str) -> str | None:
        v = d.get(key)
        return format_loss(v, dp=dp) if v is not None else None

    def pct(d: dict, key: str) -> str | None:
        v = d.get(key)
        return format_pct(v) if v is not None else None

    def rate(d: dict, key: str) -> str | None:
        v = d.get(key)
        return format_rate(v) if v is not None else None

    def val(e: dict) -> dict:
        return e.get("val") or {}

    extractors: dict[str, Extractor] = {
        "t_pol":      lambda e: loss(e, "policy"),
        "t_val":      lambda e: loss(e, "value"),
        "t_pass":     lambda e: loss(e, "pass"),
        "t_tot":      lambda e: loss(e, "total"),
        "v_pol":      lambda e: loss(val(e), "policy"),
        "v_val":      lambda e: loss(val(e), "value"),
        "v_pass":     lambda e: loss(val(e), "pass"),
        "v_tot":      lambda e: loss(val(e), "total"),
        "top1":       lambda e: pct(val(e), "top1"),
        "top3":       lambda e: pct(val(e), "top3"),
        "sps":        lambda e: rate(e, "samples_per_sec"),
        "pass_frac":  lambda e: pct(val(e), "pass_frac"),
        "pass_acc":   lambda e: pct(val(e), "pass_acc"),
    }

    registry_names = {col.name for col in _COL_REGISTRY}
    assert registry_names == set(extractors), (
        f"_COL_REGISTRY / extractors mismatch: "
        f"missing extractors {registry_names - set(extractors)}, "
        f"extra extractors {set(extractors) - registry_names}"
    )

    return [
        ColDef(
            name=col.name,
            extractor=extractors[col.name],
            long_name=LONG_NAMES.get(col.name, ""),
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

def parse_run_arg(raw: str, base: Path | None) -> tuple[Path, str]:
    if "," in raw:
        path_str, label = raw.rsplit(",", 1)
    else:
        path_str, label = raw, ""
    p = Path(path_str)
    if base is not None and not p.is_absolute() and not p.exists():
        p = base / path_str
    if not label:
        label = p.name
    return p, label


def load_epochs(run_dir: Path) -> list[dict]:
    path = run_dir / "epochs.jsonl"
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


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
                    val = col.extractor(epochs[ei])
                    run_cells.append(val if val is not None else MISSING)
            epoch_cols.append(run_cells)
        grid.append(epoch_cols)
    return max_epochs, grid


def build_table(runs: list[tuple[str, list[dict]]], cols: list[ColDef]) -> None:
    headers = ["epoch", "run"] + [col.display_name for col in cols]
    aligns = ["right", "left"] + [col.align for col in cols]

    max_epochs, grid = _extract(runs, cols)
    labels = [label for label, _ in runs]
    rows = []
    for ei in range(max_epochs):
        for ri, label in enumerate(labels):
            cells = [grid[ei][ci][ri] for ci in range(len(cols))]
            rows.append([ei + 1, label] + cells)

    print(md_table(headers, rows, align=aligns))


def build_wide_table(runs: list[tuple[str, list[dict]]], cols: list[ColDef]) -> None:
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
    print(md_table(headers, [label_row] + data_rows, align=aligns))
