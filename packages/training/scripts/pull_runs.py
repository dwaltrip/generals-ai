#!/usr/bin/env -S uv run python
"""Pull training-run artifacts off the Modal training-runs volume.

By default this mirrors the usual whole-run `modal volume get`. With
--no-checkpoints it pulls every top-level entry *except* the (large)
checkpoints/ subdir — for when you only want logs, metrics, the report, and the
profiler summary. With --skip-existing it fetches only files not already
present locally (recursing into subdirs), for cheaply topping up an existing
local copy with a resumed run's new epochs without re-downloading or --force.
Each run lands under data/training/runs-cloud/<run>/.

Example usage:
    pull_runs.py <run_name> [<run_name> ...]
    pull_runs.py --no-checkpoints <run_name> [<run_name> ...]
    pull_runs.py --skip-existing <run_name>      # top up a resumed run

Shells out to `uv run modal`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from settings import PROJECT_ROOT, RUNS_CLOUD_DIR


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


def _ls(remote: str) -> list[dict]:
    """Entries of a remote volume dir as modal's `--json` records.

    `remote` is a volume-root-relative path (the run name, or a subdir's
    `Filename` from a parent listing) — so the same call lists a run or recurses
    into a subdir."""
    out = subprocess.run(
        ["uv", "run", "modal", "volume", "ls", "--json", VOLUME, f"/{remote}"],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)


def _pull_entries(
    remote_dir: str, local_dir: Path, skip_checkpoints: bool, force: bool, skip_existing: bool
) -> None:
    """Pull one remote dir's entries into `local_dir`, honoring the skip flags.

    `skip_existing` is a *file*-level test: directories are recursed into rather
    than skipped wholesale, so a partially-present subdir (`checkpoints/`,
    `analysis/` after a resume) still fetches its new files while skipping the
    ones already pulled. Without `skip_existing`, a subdir is fetched in a single
    `modal volume get` — the cheaper path when there's nothing local to keep."""
    local_dir.mkdir(parents=True, exist_ok=True)
    for entry in _ls(remote_dir):
        name = entry["Filename"].rsplit("/", 1)[-1]
        is_dir = entry["Type"] == "dir"
        if skip_checkpoints and name == "checkpoints":
            print(f"  skip  {name}/")
        elif skip_existing and is_dir:
            _pull_entries(entry["Filename"], local_dir / name, skip_checkpoints, force, skip_existing)
        elif skip_existing and (local_dir / name).exists():
            print(f"  have  {name}")
        else:
            print(f"  pull  {name}{'/' if is_dir else ''}")
            _get(f"/{entry['Filename']}", local_dir, force)


def pull(
    run: str, dest: Path, skip_checkpoints: bool, force: bool, skip_existing: bool
) -> None:
    # `modal volume get` collapses the remote dir onto the dest path itself
    # when dest doesn't exist; with dest present it creates <dest>/<run>/.
    dest.mkdir(parents=True, exist_ok=True)
    if not skip_checkpoints and not skip_existing:
        _get(f"/{run}", dest, force)
        return

    # Per-entry walk to drop checkpoints/ and/or skip files already present. The
    # dest run dir must exist first: `modal volume get` collapses a multi-file
    # dir onto one path when the local destination is missing (it appends the
    # basename when the dest exists).
    _pull_entries(run, dest / run, skip_checkpoints, force, skip_existing)


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
        "--skip-existing", action="store_true",
        help="fetch only files absent locally (recursing into subdirs) — for "
             "topping up a resumed run without re-downloading or --force",
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
        pull(run, dest, args.no_checkpoints, args.force, args.skip_existing)


if __name__ == "__main__":
    main()
