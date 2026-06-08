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
# TODO: add --out-dir to run_bc_local.py and pass $SWEEP_DIR/runs/
#   so local sweep runs land in the sweep directory, not data/runs/.

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import itertools
import json
from pathlib import Path
import stat
import sys

from bc.train_config import TrainConfig
from utils.dict_merge import deep_merge, unflatten


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SWEEPS_DIR = REPO_ROOT / "training" / "data" / "sweeps"

_DISALLOWED_CONFIG_FIELDS = {"run_dir"}

SWEEP_SPEC_FILE = "sweep.json"
BASE_CONFIG_FILE = "base-config.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_label(value) -> str:
    """Derive a filesystem-safe label from a sweep value."""
    s = str(value)
    for ch in (" ", "/", "\\", ":"):
        s = s.replace(ch, "-")
    return s


def _validate_axes(axes: dict) -> list[tuple[str, list, list[str]]]:
    """Validate and parse axes spec. Returns [(path, values, labels), ...]
    sorted by path depth (shallowest first)."""
    parsed = []
    for axis_path, axis_spec in axes.items():
        values = axis_spec.get("values")
        if not isinstance(values, list) or len(values) == 0:
            raise SystemExit(
                f"sweep.json: axis '{axis_path}' must have a non-empty 'values' list"
            )
        labels = axis_spec.get("labels")
        if labels:
            if len(labels) != len(values):
                raise SystemExit(
                    f"sweep.json: axis '{axis_path}' has {len(values)} values "
                    f"but {len(labels)} labels"
                )
        else:
            labels = [make_label(v) for v in values]
        parsed.append((axis_path, values, labels))

    # Shallow axes first so deep_merge applies them before deeper ones.
    parsed.sort(key=lambda item: item[0].count("."))
    return parsed


def _apply_axes(
    base_config: dict,
    parsed_axes: list[tuple[str, list, list[str]]],
    combo: tuple,
) -> dict:
    """Build a resolved config by merging axis values into the base."""
    config = base_config
    for i, (axis_path, _values, _labels) in enumerate(parsed_axes):
        config = deep_merge(config, unflatten(axis_path, combo[i]))
    return config


def _generate_cells(
    base_config: dict,
    parsed_axes: list[tuple[str, list, list[str]]],
    configs_dir: Path,
) -> list[tuple[str, str]]:
    """Generate per-cell config files. Returns [(label, rel_config_path), ...]."""
    label_lists = [labels for _, _, labels in parsed_axes]
    value_lists = [values for _, values, _ in parsed_axes]

    cells = []
    for combo_labels, combo_values in zip(
        itertools.product(*label_lists),
        itertools.product(*value_lists),
        strict=True,
    ):
        label = "-".join(combo_labels)
        config = _apply_axes(base_config, parsed_axes, combo=combo_values)

        config_path = configs_dir / f"{label}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        cells.append((label, f"configs/{label}.json"))

    all_labels = [label for label, _ in cells]
    if len(set(all_labels)) != len(all_labels):
        dupes = sorted([label for label in all_labels if all_labels.count(label) > 1])
        raise SystemExit(f"sweep.json: duplicate cell labels: {dupes}")

    return cells


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

    parsed_axes = _validate_axes(axes)

    # -- validate base config --
    for field in ("manifest", "intermediate"):
        val = base_config.get(field, "")
        if not val or (isinstance(val, str) and val.startswith("<")):
            raise SystemExit(
                f"base-config.json: '{field}' is still a placeholder — fill it in"
            )

    disallowed = _DISALLOWED_CONFIG_FIELDS & base_config.keys()
    if disallowed:
        raise SystemExit(
            f"base-config.json: field(s) not allowed in sweep configs: "
            f"{', '.join(sorted(disallowed))}"
        )

    errors = TrainConfig.validate_partial(base_config)
    if errors:
        raise SystemExit(
            "base-config.json: invalid fields:\n  " + "\n  ".join(errors)
        )

    # Validate axis paths by checking a sample resolved config.
    sample_combo = tuple(values[0] for _, values, _ in parsed_axes)
    sample = _apply_axes(base_config, parsed_axes, combo=sample_combo)
    errors = TrainConfig.validate_partial(sample)
    if errors:
        raise SystemExit(
            "axis path(s) produce invalid config:\n  " + "\n  ".join(errors)
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

    cells = _generate_cells(base_config, parsed_axes, configs_dir)

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
    print("Generated run-cloud.sh, run-local.sh, fetch.sh")
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
#
# Usage: ./run-cloud.sh [extra flags for run_bc_modal.py]
#   e.g. ./run-cloud.sh --profile --skip-val --max-batches 800
set -euo pipefail

EXTRA_ARGS=("$@")

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

  cell_ok=true
  cell_output=$(uv run modal run --detach training/scripts/run_bc_modal.py \\
    --config "$SWEEP_DIR/$config" \\
    "${{EXTRA_ARGS[@]}}" \\
    2>&1) || cell_ok=false

  echo "$cell_output" | tee -a "$LAUNCH_LOG"

  if [ "$cell_ok" = false ]; then
    echo "ERROR: launch failed for $label" >&2
    echo | tee -a "$LAUNCH_LOG"
    sleep 5
    continue
  fi

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
#
# Usage: ./run-local.sh [extra flags for run_bc_local.py]
#   e.g. ./run-local.sh --profile --skip-val --max-batches 800
set -euo pipefail

EXTRA_ARGS=("$@")

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
    "${{EXTRA_ARGS[@]}}"

  echo "=== $label done ==="
  echo ""
done
"""


def render_fetch_script(sweep_rel: str, repo_root_nav: str) -> str:
    return f"""\
#!/usr/bin/env bash
# Auto-generated by prep_sweep.py — safe to regenerate.
#
# Usage: ./fetch.sh [extra flags for pull_runs.py]
#   e.g. ./fetch.sh --no-checkpoints --force
set -euo pipefail

EXTRA_ARGS=("$@")

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
  ./training/scripts/pull_runs.py --dest "$RUNS_DIR" "${{EXTRA_ARGS[@]}}" "$run_id"
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
    p_init.add_argument(
        "--axis",
        help="Dotted config path to sweep (e.g. arch.obs.dense_history_n)",
    )

    p_gen = sub.add_parser(
        "generate",
        help="Generate configs and scripts from a filled-in sweep dir",
    )
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
