from pathlib import Path


# Single source of truth for the repo root. Import PROJECT_ROOT (and paths
# derived from it) from here rather than re-deriving the root per file with
# `Path(__file__).resolve().parent...` — that depth-counting pattern silently
# breaks whenever a file moves to a different directory level.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Primary location for project data.
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# This module holds paths consumed across packages. A path used by only one
# package belongs with that package (e.g. most of training's artifact dirs live
# in `training.settings`), not in this repo-wide registry.

DB_PATH = PROJECT_ROOT / "replay-collector" / "data" / "generals.sqlite"

# Parsed-replay corpus: produced by replay-parser, consumed by training.
INTERMEDIATE_DIR = PROJECT_ROOT / "replay-parser" / "data" / "intermediate"

# Curated top-player name lists, used to filter the corpus.
CURATED_LISTS_MANIFEST = PROJECT_ROOT / "curated-player-lists.txt"

# Training-data subtree under data/. runs-cloud holds checkpoints pulled down
# from Modal; eval and self-play both load checkpoints from it, so this
# cross-package path lives here.
# Training-internal subdirs stay in `training.settings`.
TRAINING_DATA_DIR = ROOT_DATA_DIR / "training"
RUNS_CLOUD_DIR = TRAINING_DATA_DIR / "runs-cloud"

# Frozen eval map sets (manifest + packs). Cross-package: eval-tools builds and
# loads them; training's split-builder excludes their replay ids so eval maps
# stay disjoint from training data.
EVAL_MAP_SETS_DIR = ROOT_DATA_DIR / "eval" / "map-sets"
