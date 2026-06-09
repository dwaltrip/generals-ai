"""Training-owned filesystem paths.

These are artifact dirs internal to the training package. Cross-package paths
(the parsed-replay corpus, the curated lists, the cloud-runs checkpoint dir)
live in the repo-wide `settings` package instead. The training-data subtree
prefix is defined once there as `TRAINING_DATA_DIR`; the per-artifact dirs hang
off it here so each leaf is named in exactly one place.
"""

from settings import PROJECT_ROOT, TRAINING_DATA_DIR


TRAINING_DIR = PROJECT_ROOT / "packages" / "training"

# Training-run outputs. Local runs live here; cloud runs pulled from Modal land
# in `settings.RUNS_CLOUD_DIR` (cross-package — eval and self-play read it too).
RUNS_DIR = TRAINING_DATA_DIR / "runs"

# Value-head probe outputs and prepared parameter-sweep dirs.
PROBES_DIR = TRAINING_DATA_DIR / "probes"
SWEEPS_DIR = TRAINING_DATA_DIR / "sweeps"

# Pinned deps for the Modal training image (read locally at image-build time).
TRAINING_REQS = TRAINING_DIR / "modal_requirements.txt"
