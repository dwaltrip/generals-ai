"""
`bc.config` owns the training config class definitions, as well as the "stored
config block" sub-section of checkpoint dicts.
"""

# NOTE(ckpt-cfg-refactor-note): This package is in the process of absorbing the
# config classes. All new configs are here. Old ones will migrate in over time.

# TODO: StoredConfigBlock is not re-exported here, as it pulls in torch
# transitively by importing LossConfig from bc/loss.py. `bc.config` needs
# to stay torch-free for the numpy-only golden tests.
# Once LossConfig is extracted from loss.py, that problem should go away.

from training.bc.config.metrics_config import MetricsConfig
from training.bc.config.targets_config import TargetsConfig


__all__ = ["MetricsConfig", "TargetsConfig"]
