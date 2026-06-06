#!/usr/bin/env -S uv run python
"""Pull training-run artifacts off the Modal training-runs volume.

By default this mirrors the usual whole-run `modal volume get`. With
--no-checkpoints it pulls every top-level entry *except* the (large)
checkpoints/ subdir — for when you only want logs, metrics, the report, and the
profiler summary. Each run lands under training/data/runs-cloud/<run>/.

    training/scripts/pull_runs.py <run_name> [<run_name> ...]
    training/scripts/pull_runs.py --no-checkpoints <run_name> [<run_name> ...]

Shells out to `uv run modal`; run it from inside the repo so uv resolves the
workspace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


VOLUME = "generals-ai.training-runs"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEST = REPO_ROOT / "training" / "data" / "runs-cloud"


def _get(remote: str, local: Path, force: bool) -> None:
    """`modal volume get`, streaming progress to the terminal. Without `force`,
    modal aborts on any already-present file — so re-pulls need --force."""
    force_flag = ["--force"] if force else []
    subprocess.run(
        ["uv", "run", "modal", "volume", "get", *force_flag, VOLUME, remote, str(local)],
        cwd=REPO_ROOT, check=True,
    )


def _ls(run: str) -> list[dict]:
    """Top-level entries of a run dir as modal's `--json` records."""
    out = subprocess.run(
        ["uv", "run", "modal", "volume", "ls", "--json", VOLUME, f"/{run}"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)


def pull(run: str, skip_checkpoints: bool, force: bool) -> None:
    if not skip_checkpoints:
        _get(f"/{run}", DEST, force)  # modal creates runs-cloud/<run>/ under DEST
        return

    # Per-entry so we can drop checkpoints/. The dest run dir must exist first:
    # `modal volume get` collapses a multi-file dir onto one path when the local
    # destination is missing (it appends the basename when the dest exists).
    run_dest = DEST / run
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
        "--no-checkpoints", action="store_true", help="skip the checkpoints/ subdir"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing local files (modal aborts on conflicts otherwise)",
    )
    args = parser.parse_args()
    for run in args.runs:
        run = run.strip("/")
        print(f"{run} -> {DEST / run}")
        pull(run, args.no_checkpoints, args.force)


if __name__ == "__main__":
    main()
