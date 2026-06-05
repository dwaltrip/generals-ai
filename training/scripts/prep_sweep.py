#!/usr/bin/env -S uv run python
"""Scaffold and generate sweep experiments.

Two-step workflow:
  1. init  — create a sweep directory with template files to fill in.
  2. generate — read the filled-in spec and produce per-cell configs + bash scripts.

Example:
    ./training/scripts/prep_sweep.py init dense-history-n --axis arch.obs.dense_history_n
    # ... edit sweep.json and base-config.json ...
    ./training/scripts/prep_sweep.py generate training/data/sweeps/2026-06-05-dense-history-n/
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SWEEPS_DIR = REPO_ROOT / "training" / "data" / "sweeps"

SWEEP_SPEC_FILE = "sweep.json"
BASE_CONFIG_FILE = "base-config.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def deep_set(d: dict, dotted_path: str, value) -> dict:
    """Return a deep copy of *d* with *dotted_path* set to *value*."""
    result = copy.deepcopy(d)
    keys = dotted_path.split(".")
    target = result
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value
    return result


def make_label(value) -> str:
    """Derive a filesystem-safe label from a sweep value."""
    s = str(value)
    for ch in (" ", "/", "\\"):
        s = s.replace(ch, "-")
    return s


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> None:
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    sweep_dir = SWEEPS_DIR / f"{date_str}-{args.name}"

    if sweep_dir.exists():
        raise SystemExit(f"Sweep directory already exists: {sweep_dir}")

    sweep_dir.mkdir(parents=True)

    # -- sweep.json --
    axes = {}
    if args.axis:
        axes[args.axis] = {"values": [], "labels": []}

    sweep_spec = {"axes": axes}
    (sweep_dir / SWEEP_SPEC_FILE).write_text(
        json.dumps(sweep_spec, indent=2) + "\n"
    )

    # -- base-config.json --
    base_config = {
        "manifest": "<MANIFEST_PATH>",
        "intermediate": "<INTERMEDIATE_PATH>",
        "epochs": 1,
        "batch_size": 64,
        "seed": 0,
        "gpu": "H100",
        "arch": {},
    }
    (sweep_dir / BASE_CONFIG_FILE).write_text(
        json.dumps(base_config, indent=2) + "\n"
    )

    print(f"Created sweep directory: {sweep_dir}")
    print(f"  {SWEEP_SPEC_FILE}    — fill in axis values and labels")
    print(f"  {BASE_CONFIG_FILE} — fill in training config")
    print()
    print("Then run:")
    print(f"  ./training/scripts/prep_sweep.py generate {sweep_dir.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    sweep_dir = Path(args.sweep_dir).resolve()

    spec_path = sweep_dir / SWEEP_SPEC_FILE
    base_path = sweep_dir / BASE_CONFIG_FILE
    if not spec_path.exists():
        raise SystemExit(f"Missing {SWEEP_SPEC_FILE} in {sweep_dir}")
    if not base_path.exists():
        raise SystemExit(f"Missing {BASE_CONFIG_FILE} in {sweep_dir}")

    spec = json.loads(spec_path.read_text())
    base_config = json.loads(base_path.read_text())

    # -- validate spec --
    axes = spec.get("axes")
    if not isinstance(axes, dict) or len(axes) == 0:
        raise SystemExit("sweep.json: 'axes' must be a non-empty object")
    if len(axes) > 1:
        raise SystemExit(
            "sweep.json: multi-axis sweeps are not yet supported (got "
            f"{len(axes)} axes)"
        )

    axis_path, axis_spec = next(iter(axes.items()))
    values = axis_spec.get("values")
    if not isinstance(values, list) or len(values) == 0:
        raise SystemExit(
            f"sweep.json: axis '{axis_path}' must have a non-empty 'values' list"
        )

    labels = axis_spec.get("labels")
    if labels:
        if len(labels) != len(values):
            raise SystemExit(
                f"sweep.json: axis '{axis_path}' has {len(values)} values but "
                f"{len(labels)} labels"
            )
    else:
        labels = [make_label(v) for v in values]

    if len(set(labels)) != len(labels):
        raise SystemExit(f"sweep.json: duplicate labels: {labels}")

    # -- validate base config --
    for field in ("manifest", "intermediate"):
        val = base_config.get(field, "")
        if not val or (isinstance(val, str) and val.startswith("<")):
            raise SystemExit(
                f"base-config.json: '{field}' is still a placeholder — fill it in"
            )

    # -- warn if already launched --
    run_ids_path = sweep_dir / "run_ids.txt"
    if run_ids_path.exists():
        print(
            f"WARNING: {run_ids_path.name} already exists — runs may have been "
            "launched already. Regenerating configs and scripts.",
            file=sys.stderr,
        )

    # -- generate per-cell configs --
    configs_dir = sweep_dir / "configs"
    configs_dir.mkdir(exist_ok=True)

    cells = []
    for label, value in zip(labels, values):
        cell_config = deep_set(base_config, axis_path, value)
        config_path = configs_dir / f"{label}.json"
        config_path.write_text(json.dumps(cell_config, indent=2) + "\n")
        cells.append((label, f"configs/{label}.json"))

    # -- generate bash scripts --
    sweep_rel = str(sweep_dir.relative_to(REPO_ROOT))
    depth = len(Path(sweep_rel).parts)
    repo_root_nav = "/".join([".."] * depth)

    write_executable(
        sweep_dir / "run-cloud.sh",
        render_cloud_script(sweep_rel, cells, repo_root_nav),
    )
    write_executable(
        sweep_dir / "run-local.sh",
        render_local_script(sweep_rel, cells, repo_root_nav),
    )
    write_executable(
        sweep_dir / "fetch.sh",
        render_fetch_script(sweep_rel, repo_root_nav),
    )

    print(f"Generated {len(cells)} cell configs in {configs_dir.relative_to(REPO_ROOT)}/")
    print(f"Generated run-cloud.sh, run-local.sh, fetch.sh")
    print()
    print("Next steps:")
    print(f"  1. Review configs in {configs_dir.relative_to(REPO_ROOT)}/")
    print(f"  2. Launch:  ./{sweep_rel}/run-cloud.sh")
    print(f"  3. Fetch:   ./{sweep_rel}/fetch.sh")


# ---------------------------------------------------------------------------
# Bash script renderers
# ---------------------------------------------------------------------------

def render_cloud_script(sweep_rel: str, cells: list[tuple[str, str]],
                        repo_root_nav: str) -> str:
    cells_lines = "\n".join(f'  "{label}:{config_rel}"' for label, config_rel in cells)

    return f"""\
#!/usr/bin/env bash
# Auto-generated by prep_sweep.py — safe to regenerate.
set -euo pipefail

# --- operational flags (edit before running) ---
MAX_BATCHES=""
NUM_WORKERS=""

export PYTHONUNBUFFERED=1

REPO_ROOT="$(cd "$(dirname "$0")/{repo_root_nav}" && pwd)"
cd "$REPO_ROOT"

SWEEP_DIR="{sweep_rel}"
RUN_IDS_FILE="$SWEEP_DIR/run_ids.txt"
LAUNCH_LOG="$SWEEP_DIR/launch.log"
: > "$RUN_IDS_FILE"
: > "$LAUNCH_LOG"

CELLS=(
{cells_lines}
)

for entry in "${{CELLS[@]}}"; do
  IFS=":" read -r label config <<< "$entry"

  echo "=== launching $label ===" | tee -a "$LAUNCH_LOG"

  cell_output=$(uv run modal run --detach training/scripts/run_bc_modal.py \\
    --config "$SWEEP_DIR/$config" \\
    ${{MAX_BATCHES:+--max-batches "$MAX_BATCHES"}} \\
    ${{NUM_WORKERS:+--num-workers "$NUM_WORKERS"}} \\
    2>&1)

  echo "$cell_output" | tee -a "$LAUNCH_LOG"

  run_id=$(echo "$cell_output" | grep "run_dir:" | tail -1 | sed 's#.*/##' || true)
  if [ -n "$run_id" ]; then
    echo "$label $run_id" >> "$RUN_IDS_FILE"
    echo "  -> $label: $run_id"
  else
    echo "WARNING: could not capture run_id for $label" >&2
  fi

  echo | tee -a "$LAUNCH_LOG"
  sleep 5
done

echo ""
echo "All cells launched. Run IDs:"
cat "$RUN_IDS_FILE"
"""


def render_local_script(sweep_rel: str, cells: list[tuple[str, str]],
                        repo_root_nav: str) -> str:
    cells_lines = "\n".join(f'  "{label}:{config_rel}"' for label, config_rel in cells)

    return f"""\
#!/usr/bin/env bash
# Auto-generated by prep_sweep.py — safe to regenerate.
# Local runs are sequential (single GPU). No run-ID capture
# (run_bc_local.py doesn't print run_dir). Check training/data/runs/
# for output directories.
set -euo pipefail

# --- operational flags (edit before running) ---
MAX_BATCHES=""
NUM_WORKERS=""

REPO_ROOT="$(cd "$(dirname "$0")/{repo_root_nav}" && pwd)"
cd "$REPO_ROOT"

SWEEP_DIR="{sweep_rel}"

CELLS=(
{cells_lines}
)

for entry in "${{CELLS[@]}}"; do
  IFS=":" read -r label config <<< "$entry"

  echo "=== running $label ==="

  uv run training/scripts/run_bc_local.py \\
    --config "$SWEEP_DIR/$config" \\
    ${{MAX_BATCHES:+--max-batches "$MAX_BATCHES"}} \\
    ${{NUM_WORKERS:+--num-workers "$NUM_WORKERS"}}

  echo "=== $label done ==="
  echo ""
done
"""


def render_fetch_script(sweep_rel: str, repo_root_nav: str) -> str:
    return f"""\
#!/usr/bin/env bash
# Auto-generated by prep_sweep.py — safe to regenerate.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/{repo_root_nav}" && pwd)"
cd "$REPO_ROOT"

SWEEP_DIR="{sweep_rel}"
RUN_IDS_FILE="$SWEEP_DIR/run_ids.txt"
RUNS_DIR="$SWEEP_DIR/runs"

if [ ! -f "$RUN_IDS_FILE" ]; then
  echo "ERROR: $RUN_IDS_FILE not found. Run run-cloud.sh first." >&2
  exit 1
fi

mkdir -p "$RUNS_DIR"

while read -r label run_id; do
  echo "--- fetching $label ($run_id) ---"
  uv run modal volume get generals-ai.training-runs "/$run_id" "$RUNS_DIR"
  ln -sfn "$run_id" "$RUNS_DIR/$label"
done < "$RUN_IDS_FILE"

echo ""
echo "Done. Runs fetched to $RUNS_DIR/"
ls -la "$RUNS_DIR/"
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaffold and generate sweep experiments.",
    )
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Create a new sweep directory with templates")
    p_init.add_argument("name", help="Sweep name (e.g. dense-history-n)")
    p_init.add_argument("--axis", help="Dotted config path to sweep (e.g. arch.obs.dense_history_n)")

    p_gen = sub.add_parser("generate", help="Generate configs and scripts from a filled-in sweep dir")
    p_gen.add_argument("sweep_dir", help="Path to the sweep directory")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit(1)

    if args.command == "init":
        cmd_init(args)
    elif args.command == "generate":
        cmd_generate(args)


if __name__ == "__main__":
    main()
