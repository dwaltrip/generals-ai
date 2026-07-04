"""Construct live model objects from resolved config.

`build_model` is the one place a `ModelConfig` becomes a `BCModel`. Checkpoint
loading and fresh-training construction both go through it.
This module is also the named home for a future configurable builder — one that
could reconstruct historical model/obs/loss behavior from config.
"""

from __future__ import annotations

from dataclasses import dataclass

from training.bc.model import BCModel
from training.bc.model_config import ModelConfig
from training.bc.train_config import TrainConfig


def build_model(arch: ModelConfig) -> BCModel:
    return BCModel(arch)


@dataclass
class ConfiguredModel:
    """A built model paired with the config it was constructed from."""

    model: BCModel
    # NOTE(ckpt-cfg-refactor-note): Currently, config is None when a model comes
    # from a legacy checkpoint. A v0 .pt records only the "arch", while the full
    # TrainConfig was recorded in the `args.json` sidecar file.
    # A v0->v1 normalizer (not yet implemented) could reconstruct the TrainConfig.
    # But whether that normalizer would run on every load, or only on the paths
    # that need a config (like checkpoint resume), is still undecided. Only the
    # "run-on-every-load" option would let us remove `None` here.
    config: TrainConfig | None

    @property
    def cfg(self) -> ModelConfig:
        """Pass-through to the `ModelConfig` on `self.model`."""
        # NOTE(ckpt-cfg-refactor-note): the plan — deferred for now — is for
        # `ConfiguredModel` to own the `ModelConfig` and for `BCModel.cfg` to go away.
        return self.model.cfg
