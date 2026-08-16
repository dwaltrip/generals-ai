"""Read and write BC model checkpoints.

This module owns checkpoint I/O and assembly: it loads a `.pt` file into a
`ConfiguredModel` and serializes training state into the v1 layout. The
public interface is its functions — `read_*` returns a `RawCheckpoint`,
`load_*` builds models — and callers get a `RawCheckpoint` from
`read_checkpoint` rather than constructing one. `raw_checkpoint` owns the
typed view of the envelope. The stored `config` block is opaque at this
layer — `bc.config` owns its contents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from training.bc.checkpoint import arch_for_load
from training.bc.config.stored_block import resolve_config, serialize_config
from training.bc.model_builder import ConfiguredModel, build_model
from training.bc.model_config import ModelConfig
from training.bc.storage.raw_checkpoint import RawCheckpoint
from training.bc.train_config import TrainConfig


# The legacy `value_head_variant` fallback: checkpoints that predate the
# recorded arch don't say which value head they were trained with (the variant
# lived only in the run dir's `args.json`, and the files on disk are a mix of
# direct and pyramid). The caller must supply the variant for those checkpoints.
# When the checkpoint records its arch, the recorded value is authoritative
# and the argument is ignored.
def load_checkpoint(
    path: str | Path,
    device: torch.device,
    value_head_variant: str = "direct",
) -> ConfiguredModel:
    """Load a checkpoint file into a `ConfiguredModel`.

    `value_head_variant` is the legacy-only fallback (see above).
    """
    return load_from_raw(read_checkpoint(path, device), device, value_head_variant)


def read_checkpoint(path: str | Path, device: torch.device) -> RawCheckpoint:
    """Read a checkpoint file into a `RawCheckpoint`.

    Every checkpoint layout, including a bare `state_dict`, is a dict at the
    top level.
    """
    obj = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError(
            f"unrecognized checkpoint {path}: expected a dict at the top "
            f"level, got {type(obj).__name__}"
        )
    return RawCheckpoint(obj)


def load_from_raw(
    raw: RawCheckpoint,
    device: torch.device,
    value_head_variant: str = "direct",
) -> ConfiguredModel:
    """Build a `ConfiguredModel` from an already-read checkpoint.

    The entry point for callers that inspect the `RawCheckpoint` alongside the
    load. Everyone else goes through `load_checkpoint`.
    """
    if raw.versioned:
        return _load_versioned_checkpoint(raw, device)
    return _load_legacy_checkpoint(raw, device, value_head_variant)


def _load_versioned_checkpoint(raw: RawCheckpoint, device: torch.device) -> ConfiguredModel:
    """Reconstruct a model from a versioned (v1+) checkpoint.

    Returns the model on `device` in eval mode.
    """
    config = resolve_config(raw.config_block)
    model = build_model(config.arch)
    validate_in_ch(raw.in_ch, model.cfg)
    model.load_state_dict(raw.model_state)
    model.to(device).eval()
    return ConfiguredModel(model=model, config=config)


# NOTE(ckpt-cfg-refactor-note): This is mostly just a "copy" of the old `load_bc_model`
# from the pre-refactor `checkpoint.py` (in top-level `bc`).
# When we implement the proper v0 -> v1 normalizer this will shift to use that —
# or be dropped, if old checkpoints become unsupported (route-1).
def _load_legacy_checkpoint(
    raw: RawCheckpoint,
    device: torch.device,
    value_head_variant: str,
) -> ConfiguredModel:
    """Reconstruct a model from a legacy checkpoint.

    Returns the model on `device` in eval mode.
    """
    model = build_model(arch_for_load(raw.data, value_head_variant))
    model.load_state_dict(raw.model_state)
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


# NOTE(ckpt-cfg-refactor-note): the write side spells the envelope keys as
# literals here, while the read side derives facts in RawCheckpoint. The
# planned key-constants/TypedDict follow-up gives both sides one source.
def serialize_checkpoint(
    runtime: dict[str, Any], config: TrainConfig, code_sha: str
) -> dict[str, Any]:
    """Assemble a v1 checkpoint dict from runtime state, config, and provenance."""
    return {
        **runtime,
        "in_ch": config.arch.in_ch,
        "config": serialize_config(config),
        "code_sha": code_sha,
    }
