"""Read and write BC model checkpoints — the `.pt` format adapter.

This module owns the on-disk format: it reads both layouts (versioned and
legacy) into a `ConfiguredModel`, and assembles the v1 dict that gets written.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from training.bc.checkpoint import arch_for_load, is_combined_checkpoint
from training.bc.config import (
    CONFIG_VERSION,
    StoredConfigBlock,
    resolve_config,
    stringify_paths,
)
from training.bc.model_builder import ConfiguredModel, build_model
from training.bc.model_config import ModelConfig
from training.bc.train_config import TrainConfig


# TODO: Write-up $LEGACY_EXPLAINER somewhere and fill in the placeholder ref here.
def load_checkpoint(
    path: str | Path,
    device: torch.device,
    value_head_variant: str = "direct",
) -> ConfiguredModel:
    """Load a checkpoint file into a `ConfiguredModel`.

    Determine whether it is a versioned or legacy checkpoint and handle accordingly.
    `value_head_variant` is a legacy-only fallback (see $LEGACY_EXPLAINER).
    """
    obj = torch.load(path, map_location=device, weights_only=True)
    if is_versioned_checkpoint(obj):
        return _load_versioned_checkpoint(obj, device)
    return _load_legacy_checkpoint(obj, device, value_head_variant)


def is_versioned_checkpoint(obj: object) -> bool:
    """True if a loaded checkpoint carries a versioned config block."""
    return isinstance(obj, dict) and "config" in obj


def _load_versioned_checkpoint(obj: Any, device: torch.device) -> ConfiguredModel:
    """Reconstruct a model from a versioned (v1+) checkpoint.

    Returns the model on `device` in eval mode.
    """
    config = resolve_config(obj["config"])
    model = build_model(config.arch)
    validate_in_ch(obj["in_ch"], model.cfg)
    model.load_state_dict(obj["model"])
    model.to(device).eval()
    return ConfiguredModel(model=model, config=config)


# NOTE(ckpt-cfg-refactor-note): This is mostly just a "copy" of the old `load_bc_model`
# from the pre-refactor `checkpoint.py` (in top-level `bc`).
# When we implement the proper v0 -> v1 normalizer this will shift to use that —
# or be dropped, if old checkpoints become unsupported (route-1).
def _load_legacy_checkpoint(
    obj: Any,
    device: torch.device,
    value_head_variant: str,
) -> ConfiguredModel:
    """Reconstruct a model from a legacy checkpoint.

    Returns the model on `device` in eval mode.
    """
    model = build_model(arch_for_load(obj, value_head_variant))
    state_dict = obj["model"] if is_combined_checkpoint(obj) else obj
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return ConfiguredModel(model=model, config=None)


def validate_in_ch(stored_in_ch: int, cfg: ModelConfig) -> None:
    """Check a checkpoint's recorded `in_ch` against the obs channel count.

    `in_ch` records the obs-derived input-channel count as a checksum. A mismatch
    means the obs-channel formula drifted from the trained weights — caught here
    for a clear error instead of a cryptic `load_state_dict` shape mismatch.
    """
    if stored_in_ch != cfg.in_ch:
        raise ValueError(
            f"checkpoint in_ch={stored_in_ch} contradicts obs "
            f"(resolved obs config implies {cfg.in_ch} channels)"
        )


def serialize_checkpoint(
    runtime: dict[str, Any], config: TrainConfig, code_sha: str
) -> dict[str, Any]:
    """Assemble a v1 checkpoint dict from runtime state, config, and provenance."""
    block: StoredConfigBlock = {**asdict(config), "config_version": CONFIG_VERSION}
    stringify_paths(block)
    return {
        **runtime,
        "in_ch": config.arch.in_ch,
        "config": block,
        "code_sha": code_sha,
    }
