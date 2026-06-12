#!/usr/bin/env -S uv run python
"""Pull eval-run artifacts off the Modal eval-runs volume.

By default this mirrors the usual whole-run `modal volume get`. With --no-games
it pulls every top-level entry *except* the (large) games/ subdir — for when
you only want results.jsonl, the logs, and analysis/. Each run lands under
data/eval/runs-cloud/<run>/.

Example usage:
    pull_eval_runs.py <run_name> [<run_name> ...]
    pull_eval_runs.py --no-games <run_name> [<run_name> ...]

Shells out to `uv run modal`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from settings import PROJECT_ROOT, ROOT_DATA_DIR


VOLUME = "generals-ai.eval-runs"
DEST = ROOT_DATA_DIR / "eval" / "runs-cloud"
# Run dirs sit under runs/ at the volume root (map-sets/ is the other subtree).
REMOTE_RUNS_PREFIX = "runs"


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
        ["uv", "run", "modal", "volume", "ls", "--json", VOLUME,
         f"/{REMOTE_RUNS_PREFIX}/{run}"],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)


def pull(run: str, dest: Path, skip_games: bool, force: bool) -> None:
    if not skip_games:
        # modal creates <dest>/<run>/ under dest (basename of the remote path)
        _get(f"/{REMOTE_RUNS_PREFIX}/{run}", dest, force)
        return

    # Per-entry so we can drop games/. The dest run dir must exist first:
    # `modal volume get` collapses a multi-file dir onto one path when the local
    # destination is missing (it appends the basename when the dest exists).
    run_dest = dest / run
    run_dest.mkdir(parents=True, exist_ok=True)
    for entry in _ls(run):
        name = entry["Filename"].rsplit("/", 1)[-1]
        if name == "games":
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
        "--no-games", action="store_true", help="skip the games/ subdir"
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
        pull(run, dest, args.no_games, args.force)


if __name__ == "__main__":
    main()
