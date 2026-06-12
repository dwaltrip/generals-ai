"""Checkpoint I/O for BC models.

The single place inference and probe code loads model weights from a
`.pt` file, and the home for the on-disk checkpoint format knowledge
(filename scheme + the combined-vs-legacy layout), so that knowledge
lives in one module rather than being duplicated across call sites.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, TypeGuard

import torch

from training.bc.model import BCModel
from training.bc.model_config import ModelConfig, build_model_cfg
from training.bc.obs_config import ObsConfig


# Historical facts about on-disk checkpoints, NOT live defaults. DO NOT EDIT.
# Kept separate from the live defaults on purpose: the day we change the default
# width or dense_history_n, aliasing legacy = MODEL_CONFIG_DEFAULTS / OBS_CONFIG_DEFAULTS
# would silently mis-load every old checkpoint. They're equal today and allowed
# to diverge — hence the literals here rather than default references.
#
# LEGACY_OBS_CFG is the single home for the historical obs contract: the
# pre-`obs`-key dense-history depth AND the pre-`obs_dtype`-field element dtype
# (always fp32 before that field existed). It backs LEGACY_ARCH.obs (pre-`arch`
# checkpoints), the `_arch_for_load` whole-`obs` fill (arch-bearing-but-pre-`obs`
# checkpoints), and the per-sub-key back-fill (checkpoints with `obs` but missing
# newer keys like `obs_dtype`). `obs_dtype="fp32"` here is the first field where
# this diverges from the live default — recording history, not editing a value.
LEGACY_OBS_CFG = ObsConfig(dense_history_n=5, obs_dtype="fp32")
LEGACY_ARCH = ModelConfig(
    outer_width=128, middle_width=128, inner_width=160,
    n_outer=2, m_middle=2, m_inner=2,
    value_head_variant="direct", H=32, W=32,
    # Pre-field behavior: no head dropout existed before these fields did.
    value_head_dropout2d=0.0, value_head_dropout=0.0,
    # Pre-field behavior: dropout2d (where used) sat after `pre`; skip
    # connections were never dropped.
    value_head_dropout2d_site="post_pre", value_head_skip_dropout2d=0.0,
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

    Any key missing from a recorded arch dict predates that field (saves are
    fully-resolved `asdict`s), so missing keys back-fill from the LEGACY pins,
    not the live defaults — filling from live defaults would silently
    re-describe old checkpoints the day a default changes (e.g. an fp32-obs
    checkpoint loaded as fp16, or a no-dropout checkpoint resumed with
    dropout). `build_model_cfg` / `ModelConfig.__post_init__` merge from the
    live defaults, so the legacy back-fill happens here, before they run.
    Two granularities, same rule: top-level keys fill from `LEGACY_ARCH`,
    obs sub-keys from `LEGACY_OBS_CFG` (a checkpoint missing `obs` entirely
    gets the whole legacy obs config via the top-level fill).
    """
    if is_combined_checkpoint(obj) and "arch" in obj:
        recorded = dict(obj["arch"])
        obs = recorded.get("obs")
        arch_dict = {**asdict(LEGACY_ARCH), **recorded}
        if isinstance(obs, dict):
            arch_dict["obs"] = {**asdict(LEGACY_OBS_CFG), **obs}
        # in_ch is a recorded checksum (TrainingState.save), derived from obs.
        # Validate it here — the one place a count baked into trained weights
        # meets a possibly-changed obs-channel formula — for a clear error rather
        # than a cryptic state_dict shape mismatch.
        stored_in_ch = arch_dict.pop("in_ch", None)
        cfg = build_model_cfg(**arch_dict)
        if stored_in_ch is not None and stored_in_ch != cfg.in_ch:
            raise ValueError(
                f"checkpoint in_ch={stored_in_ch} contradicts obs "
                f"(dense_history_n={cfg.obs.dense_history_n} → {cfg.in_ch} channels)"
            )
        return cfg
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
