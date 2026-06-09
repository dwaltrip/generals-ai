"""Training-owned filesystem paths.

These are artifact dirs internal to the training package. Cross-package paths
(the parsed-replay corpus, the curated lists) live in the repo-wide `settings`
package instead. Everything here hangs off `TRAINING_DIR` so the
`packages/training` prefix is defined in exactly one place — when the package
moves, only this constant changes.
"""

from settings import PROJECT_ROOT


TRAINING_DIR = PROJECT_ROOT / "packages" / "training"
TRAINING_DATA = TRAINING_DIR / "data"

# Training-run outputs: local runs, and cloud runs pulled down from Modal.
RUNS_DIR = TRAINING_DATA / "runs"
RUNS_CLOUD_DIR = TRAINING_DATA / "runs-cloud"

# Value-head probe outputs and prepared parameter-sweep dirs.
PROBES_DIR = TRAINING_DATA / "probes"
SWEEPS_DIR = TRAINING_DATA / "sweeps"

# Pinned deps for the Modal training image (read locally at image-build time).
TRAINING_REQS = TRAINING_DIR / "modal_requirements.txt"
