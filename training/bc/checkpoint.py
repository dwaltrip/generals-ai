"""Checkpoint I/O for BC models.

The single place inference and probe code loads model weights from a
`.pt` file, and the home for the on-disk checkpoint format knowledge
(filename scheme + the combined-vs-legacy layout), so that knowledge
lives in one module rather than being duplicated across call sites.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, TypeGuard

import torch

from bc.model import BCModel
from bc.model_config import ModelConfig
from bc.obs_config import ObsConfig


# Historical facts about on-disk checkpoints, NOT live defaults. DO NOT EDIT.
# Kept separate from the live defaults on purpose: the day we change the default
# width or dense_history_n, aliasing legacy = ModelConfig() / OBS_CONFIG_DEFAULTS
# would silently mis-load every old checkpoint. They're equal today and allowed
# to diverge — hence the literals here rather than default references.
#
# LEGACY_OBS_CFG is the single home for the pre-`obs`-key dense-history depth: it
# backs both LEGACY_ARCH.obs (pre-`arch` checkpoints) and the `_arch_for_load`
# fill (arch-bearing-but-pre-`obs` checkpoints, e.g. the early width sweep).
LEGACY_OBS_CFG = ObsConfig(dense_history_n=5)
LEGACY_ARCH = ModelConfig(
    outer_width=128, middle_width=128, inner_width=160,
    n_outer=2, m_middle=2, m_inner=2,
    in_ch=96, H=32, W=32,
    obs=LEGACY_OBS_CFG,
)


def ckpt_name(epoch: int) -> str:
    """Deterministic checkpoint filename for an epoch.

    Split from the save itself so callers can refer to the planned name
    in logs or the `epochs.jsonl` record *before* the save executes —
    used to preserve epoch metrics if the save itself raises.
    """
    return f"epoch_{epoch:03d}.pt"


def is_combined_checkpoint(obj: object) -> TypeGuard[dict[str, Any]]:
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


def is_arch_bearing(path: str | Path, device: str | torch.device = "cpu") -> bool:
    """True if the checkpoint records its architecture (the `arch` key).

    Arch-bearing checkpoints reconstruct authoritatively, so the load-time
    `value_head_variant` arg is ignored — it must not enter the inference
    `model_key` / handle-cache key. A combined checkpoint written before the
    `arch` key existed is *not* arch-bearing and falls back to `LEGACY_ARCH`.
    """
    obj = torch.load(path, map_location=device, weights_only=True)
    return is_combined_checkpoint(obj) and "arch" in obj


def _arch_for_load(obj: object, value_head_variant: str) -> ModelConfig:
    """The `ModelConfig` to reconstruct a loaded checkpoint with.

    `arch` present → authoritative (the load-time `value_head_variant` arg is
    ignored). Absent (legacy) → `LEGACY_ARCH` widths + the load-time variant,
    since legacy checkpoints don't record their variant (it lived in the run
    dir's `args.json`, and the ones on disk are a mix of direct/pyramid).

    Arch dicts written before the `obs` key existed get `LEGACY_OBS_CFG` filled
    in (those checkpoints were all dense_history_n=5) — pinned to the historical
    value, not the live default, so a future re-default doesn't re-describe them.
    """
    if is_combined_checkpoint(obj) and "arch" in obj:
        arch_dict = dict(obj["arch"])
        arch_dict.setdefault("obs", LEGACY_OBS_CFG)
        return ModelConfig(**arch_dict)
    return replace(LEGACY_ARCH, value_head_variant=value_head_variant)


def load_bc_model(
    path: str | Path,
    device: torch.device,
    value_head_variant: str = "direct",
) -> BCModel:
    """Construct a BCModel and load weights from a `.pt` checkpoint.

    Handles both checkpoint layouts: the combined dict written by
    `TrainingState.save` (`{"model": ..., "arch": ..., ...}`) and the legacy
    bare `state_dict` (a flat map of parameter tensors). The two are
    distinguished by the presence of a top-level `"model"` key — a bare
    state_dict's keys are parameter names like `trunk.0.weight`.

    Architecture comes from the checkpoint's `arch` key when present; otherwise
    `LEGACY_ARCH` + the `value_head_variant` arg (a legacy-only fallback,
    ignored for arch-bearing checkpoints). `load_state_dict(strict=True)`
    remains the backstop: any arch↔weights drift still raises on mismatched
    keys. Returns the model on `device` in eval mode.
    """
    obj = torch.load(path, map_location=device, weights_only=True)
    model = BCModel(_arch_for_load(obj, value_head_variant))
    state_dict = obj["model"] if is_combined_checkpoint(obj) else obj
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
