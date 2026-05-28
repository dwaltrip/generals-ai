"""Checkpoint I/O for BC models.

The single place inference and probe code loads model weights from a
`.pt` file, so format knowledge lives in one helper rather than being
duplicated at each call site.
"""

from __future__ import annotations

from pathlib import Path

import torch

from bc.model import BCModel


def load_bc_model(
    path: str | Path,
    device: torch.device,
    value_head_variant: str = "direct",
) -> BCModel:
    """Construct a BCModel and load weights from a `.pt` state-dict file.

    `value_head_variant` must match the variant the checkpoint was trained
    with — otherwise `load_state_dict(strict=True)` raises on mismatched
    keys, surfacing checkpoint/architecture drift loudly. Returns the model
    on `device` in eval mode.
    """
    model = BCModel(value_head_variant=value_head_variant)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
