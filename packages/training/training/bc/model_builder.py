"""Construct live model objects from resolved config.

The seam between a `ModelConfig` and the `BCModel` it specifies, shared by the
checkpoint loader and fresh-training construction. Isolating construction here
gives a future configurable builder — reconstructing historical model/obs/loss
behavior from config — one home, instead of scattering `BCModel(...)` calls
across loaders.
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
    # NOTE(ckpt-cfg-refactor-note): None for legacy checkpoints — a v0 .pt records only the
    # arch (reachable via `cfg`), not the full TrainConfig (the recipe lived in the
    # run-dir sidecar). A later v0->v1 normalizer may reconstruct one.
    # Whether `None` goes away depends on running it on every load ("uniform") vs. only
    # on demand like the "checkpoint resume" path ("graded" — None stays). Open fork.
    config: TrainConfig | None = None

    @property
    def cfg(self) -> ModelConfig:
        """Pass-through to the `ModelConfig` on `self.model`."""
        # NOTE(ckpt-cfg-refactor-note): The current plan is to remove `.cfg` from BCModel
        # and have `ConfiguredModel` own it instead. We are deferring that for now.
        return self.model.cfg
