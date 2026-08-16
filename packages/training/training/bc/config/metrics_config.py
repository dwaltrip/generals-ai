"""MetricsConfig: measurement knobs for a training run.

This sub-config owns tangential configs that have no impact on model weights
and are only used to configure other recorded artifacts (e.g. val dumps).
"""

# NOTE: This is torch-free, as it is imported by the numpy-only golden tests.

# NOTE: The concept behind MetricsConfig is new. We are starting with a
# deliberately minimal, incomplete form for the goldens first cut work.
# The shape may shift while the concept settles.
# The class should grow into the full "measurement" sub-config afterward.
# See some design thoughts from session note 8.15-2.

# TODO(config-bump): This isn't yet implemented as an actual sub-config on
# TrainConfig. E.g. A run could not currently indicate that it wants
# `alive_mask` included in the dump unequivocally.
# Doing so requires a CONFIG_VERSION bump. We are deferring that until after
# the first cut is done (most likely).

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from training.bc.train_config import TrainConfig

@dataclass(frozen=True, kw_only=True)
class MetricsConfig:
    include_alive_mask: bool  # request the alive_mask column in samples/dumps


def metrics_cfg_from(train_cfg: TrainConfig) -> MetricsConfig:
    # TODO(config-bump): Move this derivation into MIGRATIONS[1] as the historical fill.
    # At that time, we should re-consider what an appropriate builder function may be.
    return MetricsConfig(include_alive_mask=train_cfg.arch.elim_head_variant is not None)
