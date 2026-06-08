from pathlib import Path


# Single source of truth for the repo root. Import PROJECT_ROOT (and paths
# derived from it) from here rather than re-deriving the root per file with
# `Path(__file__).resolve().parent...` — that depth-counting pattern silently
# breaks whenever a file moves to a different directory level.
# TODO: migrate the remaining per-file `Path(__file__).resolve().parents[...]`
# root computations across scripts/ and tests/ onto this PROJECT_ROOT.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DB_PATH = PROJECT_ROOT / "replay-collector" / "data" / "generals.sqlite"
