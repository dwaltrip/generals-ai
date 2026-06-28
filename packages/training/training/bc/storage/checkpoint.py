"""Load BC model checkpoints from disk.

Read `.pt` file, discriminate the on-disk layout, reconstruct `BCModel`, and
return that wrapped in a `ConfiguredModel`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from training.bc.checkpoint import arch_for_load, is_combined_checkpoint
from training.bc.model_builder import ConfiguredModel, build_model


# TODO: Write-up $LEGACY_EXPLAINER somewhere and fill in the placeholder ref here.
def load_checkpoint(
    path: str | Path,
    device: torch.device,
    value_head_variant: str = "direct",
) -> ConfiguredModel:
    """Load a checkpoint file into a `ConfiguredModel`.

    Discriminate if it contains a post-refactor "stamped, versioned config" or not,
    and pass to the appropriate handler.
    `value_head_variant` is a legacy-only fallback (see $LEGACY_EXPLAINER).
    """
    obj = torch.load(path, map_location=device, weights_only=True)
    if is_versioned_checkpoint(obj):
        raise NotImplementedError("versioned read path not yet implemented")
    return _load_legacy_checkpoint(obj, device, value_head_variant)


def is_versioned_checkpoint(obj: object) -> bool:
    """True if a loaded checkpoint carries a self-describing versioned config block."""
    # TODO: detect the versioned layout (`"config" in obj`) once the writer emits it.
    return False


# NOTE(refactor-note): This is mostly just a "copy" of the old `load_bc_model`
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
