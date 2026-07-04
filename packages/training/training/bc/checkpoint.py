"""Legacy checkpoint reconstruction and shared `.pt` helpers.

Holds the knowledge that keeps pre-versioned checkpoints loadable: the frozen
historical pins and the v0 arch reconstruction. Also home to the
combined/legacy discriminators and the filename scheme. `bc.storage.checkpoint`
delegates here for its legacy read path.
"""

# NOTE(ckpt-cfg-refactor-note): This module is superseded by
# `bc/storage/checkpoint.py`. As the refactor proceeds, whatever is still
# needed here moves there (or into a v0->v1 normalizer) and this file goes
# away. Comments and docstrings here predate the refactor and haven't been
# swept — some describe the pre-v1 world (e.g. where `in_ch` gets written).

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, TypeGuard

import torch

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
# checkpoints), the `arch_for_load` whole-`obs` fill (arch-bearing-but-pre-`obs`
# checkpoints), and the per-sub-key back-fill (checkpoints with `obs` but missing
# newer keys like `obs_dtype`). `obs_dtype="fp32"` and `player_status_channels=
# False` here diverge from the live defaults — recording history (pre-status
# checkpoints had neither the fp16 obs nor the status channels), not editing a
# value. The back-fill gives an old checkpoint these legacy values, so it
# reconstructs the original obs (no status channels) and loads unchanged.
LEGACY_OBS_CFG = ObsConfig(
    dense_history_n=5, obs_dtype="fp32", player_status_channels=False
)
LEGACY_ARCH = ModelConfig(
    outer_width=128, middle_width=128, inner_width=160,
    n_outer=2, m_middle=2, m_inner=2,
    value_head_variant="direct", H=32, W=32,
    # Pre-field behavior: no head dropout existed before these fields did.
    value_head_dropout2d=0.0, value_head_dropout=0.0,
    # Pre-field behavior: dropout2d (where used) sat after `pre`; skip
    # connections were never dropped.
    value_head_dropout2d_site="post_pre", value_head_skip_dropout2d=0.0,
    # Pre-field behavior: no elim head existed. Variant/edges/pool/hidden are
    # hardcoded literals (NOT aliased to MODEL_CONFIG_DEFAULTS, per the
    # divergence-allowed contract above) — a pre-elim checkpoint back-fills these
    # and reconstructs as "no elim head" (`elim_head_variant=None`), so
    # `load_state_dict(strict=True)` still passes. Checkpoints from the
    # `elim_head_enabled` era record that removed key instead of
    # `elim_head_variant`; `arch_for_load` migrates them before this back-fill
    # runs.
    elim_head_variant=None,
    elim_bin_edges=(10, 20, 40, 80, 160, 320, 640),
    elim_pool="mean", elim_head_hidden=0,
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


def arch_for_load(obj: object, value_head_variant: str) -> ModelConfig:
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
        ckpt_arch = dict(obj["arch"])
        # Migrate the retired `elim_head_enabled` flag to `elim_head_variant`.
        # Every checkpoint that recorded an enabled elim head was the time_bin
        # head (the only variant that existed then); disabled → no head (None).
        # Drop the removed key so it can't reach `build_model_cfg`/`replace()`,
        # and only synthesize a variant if the checkpoint didn't already record
        # one (a post-migration checkpoint records `elim_head_variant` directly).
        if "elim_head_enabled" in ckpt_arch:
            was_enabled = ckpt_arch.pop("elim_head_enabled")
            ckpt_arch.setdefault(
                "elim_head_variant", "time_bin" if was_enabled else None
            )
        obs = ckpt_arch.get("obs")
        arch_dict = {**asdict(LEGACY_ARCH), **ckpt_arch}
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
                f"(resolved obs config implies {cfg.in_ch} channels)"
            )
        return cfg
    return replace(LEGACY_ARCH, value_head_variant=value_head_variant)
