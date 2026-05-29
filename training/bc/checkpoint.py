"""Checkpoint I/O for BC models.

The single place inference and probe code loads model weights from a
`.pt` file, and the home for the on-disk checkpoint format knowledge
(filename scheme + the combined-vs-legacy layout), so that knowledge
lives in one module rather than being duplicated across call sites.
"""

from __future__ import annotations

from pathlib import Path

import torch

from bc.model import BCModel


def ckpt_name(epoch: int) -> str:
    """Deterministic checkpoint filename for an epoch.

    Split from the save itself so callers can refer to the planned name
    in logs or the `epochs.jsonl` record *before* the save executes —
    used to preserve epoch metrics if the save itself raises.
    """
    return f"epoch_{epoch:03d}.pt"


def is_combined_checkpoint(obj: object) -> bool:
    """True if a loaded checkpoint object is the combined dict format.

    The combined format (written by `TrainingState.save`) is a dict carrying a
    top-level `"model"` key alongside `optim`/`scaler`/`epoch`. A legacy bare
    `state_dict` is also a dict, but its keys are parameter names
    (`trunk.0.weight`, ...), so the `"model"` key is the discriminator.
    """
    return isinstance(obj, dict) and "model" in obj


def is_legacy_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> bool:
    """True if `path` is a legacy bare-`state_dict` checkpoint.

    Reads the checkpoint to inspect its top-level structure. The resume path
    uses this to decide whether a cold optimizer restart (and the
    `--legacy-lr-warmup-batches` opt-in) is required. Loads to CPU by default —
    only the structure is needed, not a device-resident model.
    """
    obj = torch.load(path, map_location=device, weights_only=True)
    return not is_combined_checkpoint(obj)


def load_bc_model(
    path: str | Path,
    device: torch.device,
    value_head_variant: str = "direct",
) -> BCModel:
    """Construct a BCModel and load weights from a `.pt` checkpoint.

    Handles both checkpoint layouts: the combined dict written by
    `TrainingState.save` (`{"model": ..., "optim": ..., ...}`) and the
    legacy bare `state_dict` (a flat map of parameter tensors). The two
    are distinguished by the presence of a top-level `"model"` key — a
    bare state_dict's keys are parameter names like `trunk.0.weight`.

    `value_head_variant` must match the variant the checkpoint was trained
    with — otherwise `load_state_dict(strict=True)` raises on mismatched
    keys, surfacing checkpoint/architecture drift loudly. Returns the model
    on `device` in eval mode.
    """
    model = BCModel(value_head_variant=value_head_variant)
    obj = torch.load(path, map_location=device, weights_only=True)
    state_dict = obj["model"] if is_combined_checkpoint(obj) else obj
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
