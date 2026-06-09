#!/usr/bin/env -S uv run python
"""Pull training-run artifacts off the Modal training-runs volume.

By default this mirrors the usual whole-run `modal volume get`. With
--no-checkpoints it pulls every top-level entry *except* the (large)
checkpoints/ subdir — for when you only want logs, metrics, the report, and the
profiler summary. Each run lands under packages/training/data/runs-cloud/<run>/.

Example usage:
    pull_runs.py <run_name> [<run_name> ...]
    pull_runs.py --no-checkpoints <run_name> [<run_name> ...]

Shells out to `uv run modal`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from settings import PROJECT_ROOT
from training.settings import RUNS_CLOUD_DIR


VOLUME = "generals-ai.training-runs"
DEST = RUNS_CLOUD_DIR


def _get(remote: str, local: Path, force: bool) -> None:
    """`modal volume get`, streaming progress to the terminal. Without `force`,
    modal aborts on any already-present file — so re-pulls need --force."""
    force_flag = ["--force"] if force else []
    subprocess.run(
        ["uv", "run", "modal", "volume", "get", *force_flag, VOLUME, remote, str(local)],
        cwd=PROJECT_ROOT, check=True,
    )


def _ls(run: str) -> list[dict]:
    """Top-level entries of a run dir as modal's `--json` records."""
    out = subprocess.run(
        ["uv", "run", "modal", "volume", "ls", "--json", VOLUME, f"/{run}"],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)


def pull(run: str, dest: Path, skip_checkpoints: bool, force: bool) -> None:
    if not skip_checkpoints:
        _get(f"/{run}", dest, force)  # modal creates <dest>/<run>/ under dest
        return

    # Per-entry so we can drop checkpoints/. The dest run dir must exist first:
    # `modal volume get` collapses a multi-file dir onto one path when the local
    # destination is missing (it appends the basename when the dest exists).
    run_dest = dest / run
    run_dest.mkdir(parents=True, exist_ok=True)
    for entry in _ls(run):
        name = entry["Filename"].rsplit("/", 1)[-1]
        if name == "checkpoints":
            print(f"  skip  {name}/")
            continue
        print(f"  pull  {name}{'/' if entry['Type'] == 'dir' else ''}")
        _get(f"/{entry['Filename']}", run_dest, force)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="+", help="run dir name(s) on the volume")
    parser.add_argument(
        "--dest", type=Path, default=DEST,
        help=f"local directory to pull into (default: {DEST.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--no-checkpoints", action="store_true", help="skip the checkpoints/ subdir"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing local files (modal aborts on conflicts otherwise)",
    )
    args = parser.parse_args()
    dest = args.dest.resolve()
    for run in args.runs:
        run = run.strip("/")
        print(f"{run} -> {dest / run}")
        pull(run, dest, args.no_checkpoints, args.force)


if __name__ == "__main__":
    main()
