from pathlib import Path


# Single source of truth for the repo root. Import PROJECT_ROOT (and paths
# derived from it) from here rather than re-deriving the root per file with
# `Path(__file__).resolve().parent...` — that depth-counting pattern silently
# breaks whenever a file moves to a different directory level.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Primary location for project data
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# Paths shared across packages live here. Single-package paths belong with
# the package that owns them (e.g. training's artifact dirs in
# `training.settings`), not in this repo-wide registry.
DB_PATH = PROJECT_ROOT / "replay-collector" / "data" / "generals.sqlite"

# Parsed-replay corpus: produced by replay-parser, consumed by training.
INTERMEDIATE_DIR = PROJECT_ROOT / "replay-parser" / "data" / "intermediate"

# Curated top-player name lists, used to filter the corpus.
CURATED_LISTS_MANIFEST = PROJECT_ROOT / "curated-player-lists.txt"
