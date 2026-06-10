"""`load_bc_model` reconstructs a model's architecture from the checkpoint:
the `arch` key when present (authoritative), `LEGACY_ARCH` + the load-time
variant arg for legacy bare-state_dict checkpoints. Plus the combined-vs-legacy
format detection used by the resume path."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest
import torch

from training.bc.checkpoint import (
    LEGACY_ARCH,
    LEGACY_OBS_CFG,
    is_legacy_checkpoint,
    load_bc_model,
)
from training.bc.model import BCModel
from training.bc.model_config import MODEL_CONFIG_DEFAULTS, build_model_cfg
from training.bc.obs_config import OBS_CONFIG_DEFAULTS


def test_load_bc_model_bare_state_dict(tmp_path):
    """A legacy bare state_dict (no `arch` key) reconstructs via LEGACY_ARCH."""
    device = torch.device("cpu")
    src = BCModel()
    ckpt = tmp_path / "epoch_001.pt"
    torch.save(src.state_dict(), ckpt)

    loaded = load_bc_model(ckpt, device)

    assert isinstance(loaded, BCModel)
    assert loaded.cfg == LEGACY_ARCH       # legacy fallback, default variant
    assert not loaded.training             # returned in eval mode
    src_sd, loaded_sd = src.state_dict(), loaded.state_dict()
    assert src_sd.keys() == loaded_sd.keys()
    for k in src_sd:
        assert torch.equal(src_sd[k], loaded_sd[k].cpu()), f"param mismatch: {k}"


def test_load_legacy_honors_value_head_variant_arg(tmp_path):
    """A legacy pyramid-trained checkpoint loads only when the load-time variant
    matches — strict load is the backstop. This is the unit-level twin of the
    parity fingerprint (which loads a real pyramid legacy checkpoint)."""
    device = torch.device("cpu")
    src = BCModel(build_model_cfg(value_head_variant="pyramid"))
    ckpt = tmp_path / "epoch_001.pt"
    torch.save(src.state_dict(), ckpt)  # bare state_dict — legacy

    loaded = load_bc_model(ckpt, device, value_head_variant="pyramid")
    assert loaded.cfg == replace(LEGACY_ARCH, value_head_variant="pyramid")


def test_arch_bearing_checkpoint_reconstructs_nondefault(tmp_path):
    """The round-trip the refactor exists for: a non-default arch is recorded in
    the checkpoint's `arch` key and reconstructed authoritatively on load (the
    load-time variant arg is ignored), with strict weight load."""
    device = torch.device("cpu")
    cfg = build_model_cfg(outer_width=64, middle_width=64, inner_width=96)
    src = BCModel(cfg)
    ckpt = tmp_path / "epoch_002.pt"
    # Mimic TrainingState.save's combined format (model + arch).
    torch.save({"model": src.state_dict(), "arch": asdict(cfg), "epoch": 2}, ckpt)

    loaded = load_bc_model(ckpt, device, value_head_variant="direct")
    assert loaded.cfg == cfg               # arch key authoritative
    src_sd, loaded_sd = src.state_dict(), loaded.state_dict()
    assert src_sd.keys() == loaded_sd.keys()
    for k in src_sd:
        assert torch.equal(src_sd[k], loaded_sd[k].cpu()), f"param mismatch: {k}"


def test_arch_in_ch_checksum_mismatch_raises(tmp_path):
    """A recorded in_ch contradicting the obs channel count (an obs-channel-
    formula drift) raises a clear error on load, not a cryptic shape mismatch."""
    cfg = build_model_cfg()
    src = BCModel(cfg)
    ckpt = tmp_path / "epoch_001.pt"
    arch = {**asdict(cfg), "in_ch": cfg.in_ch + 1}  # tampered checksum
    torch.save({"model": src.state_dict(), "arch": arch, "epoch": 1}, ckpt)

    with pytest.raises(ValueError, match="contradicts obs"):
        load_bc_model(ckpt, torch.device("cpu"))


def test_default_arch_is_no_op_against_bare_model(tmp_path):
    """`build_model_cfg()` yields the canonical default (`MODEL_CONFIG_DEFAULTS`),
    so a bare `BCModel()` and one built from the factory default match (the smoke
    fingerprint pins the values)."""
    assert build_model_cfg() == MODEL_CONFIG_DEFAULTS
    assert BCModel().state_dict().keys() == BCModel(build_model_cfg()).state_dict().keys()


def test_is_legacy_checkpoint_detection(tmp_path):
    legacy = tmp_path / "legacy.pt"
    combined = tmp_path / "combined.pt"
    torch.save(BCModel().state_dict(), legacy)            # bare state_dict
    torch.save({"model": {}, "optim": {}, "epoch": 3}, combined)  # combined wrapper

    assert is_legacy_checkpoint(legacy) is True
    assert is_legacy_checkpoint(combined) is False


# --- obs sub-config back-fill: load → legacy, fresh → live default ---------
#
# When a new ObsConfig field is added (obs_dtype is the first), a checkpoint's
# `obs` block predates it. The load path must back-fill the missing key from the
# FROZEN legacy snapshot, while fresh construction fills from the LIVE defaults.
# These tests force `DEFAULTS != LEGACY` so they discriminate regardless of what
# the current live default happens to be.


def test_load_missing_obs_dtype_uses_legacy(tmp_path, monkeypatch):
    """A checkpoint whose `obs` block lacks `obs_dtype` back-fills it from the
    legacy snapshot, not the live default — so flipping the default never
    silently re-describes an old checkpoint's obs precision."""
    monkeypatch.setattr(
        "training.bc.model_config.OBS_CONFIG_DEFAULTS",
        replace(OBS_CONFIG_DEFAULTS, obs_dtype="fp16"),
    )
    cfg = build_model_cfg()
    src = BCModel(cfg)
    arch = asdict(cfg)
    del arch["obs"]["obs_dtype"]  # simulate a pre-obs_dtype checkpoint
    ckpt = tmp_path / "epoch_001.pt"
    torch.save({"model": src.state_dict(), "arch": arch, "epoch": 1}, ckpt)

    loaded = load_bc_model(ckpt, torch.device("cpu"))
    assert loaded.cfg.obs.obs_dtype == LEGACY_OBS_CFG.obs_dtype  # legacy, not the fp16 default


def test_load_preserves_explicit_obs_dtype(tmp_path):
    """An `obs_dtype` explicitly recorded in a checkpoint's arch survives load —
    neither the legacy back-fill nor the default merge overrides it."""
    cfg = build_model_cfg()
    src = BCModel(cfg)
    arch = asdict(cfg)
    arch["obs"]["obs_dtype"] = "fp16"
    ckpt = tmp_path / "epoch_001.pt"
    torch.save({"model": src.state_dict(), "arch": arch, "epoch": 1}, ckpt)

    loaded = load_bc_model(ckpt, torch.device("cpu"))
    assert loaded.cfg.obs.obs_dtype == "fp16"


def test_load_missing_toplevel_key_uses_legacy(tmp_path, monkeypatch):
    """The top-level twin of the obs_dtype test: an arch dict predating a
    top-level ModelConfig field (value_head_dropout is the first such
    addition) back-fills it from LEGACY_ARCH, not the live default."""
    monkeypatch.setattr(
        "training.bc.model_config.MODEL_CONFIG_DEFAULTS",
        replace(MODEL_CONFIG_DEFAULTS, value_head_dropout=0.5),
    )
    cfg = build_model_cfg(value_head_dropout=0.0)
    src = BCModel(cfg)
    arch = asdict(cfg)
    del arch["value_head_dropout"]    # simulate a pre-field checkpoint
    del arch["value_head_dropout2d"]
    ckpt = tmp_path / "epoch_001.pt"
    torch.save({"model": src.state_dict(), "arch": arch, "epoch": 1}, ckpt)

    loaded = load_bc_model(ckpt, torch.device("cpu"))
    assert loaded.cfg.value_head_dropout == LEGACY_ARCH.value_head_dropout  # 0.0, not 0.5


def test_fresh_partial_obs_uses_live_default(monkeypatch):
    """A fresh config from a partial `obs` dict (not a checkpoint load) fills
    `obs_dtype` from the live defaults — the complement of the load-path
    back-fill above."""
    monkeypatch.setattr(
        "training.bc.model_config.OBS_CONFIG_DEFAULTS",
        replace(OBS_CONFIG_DEFAULTS, obs_dtype="fp16"),
    )
    cfg = build_model_cfg(obs={"dense_history_n": 5})
    assert cfg.obs.obs_dtype == "fp16"  # live default, not legacy fp32
