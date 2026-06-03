"""`load_bc_model` reconstructs a model's architecture from the checkpoint:
the `arch` key when present (authoritative), `LEGACY_ARCH` + the load-time
variant arg for legacy bare-state_dict checkpoints. Plus the combined-vs-legacy
format detection used by the resume path."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest
import torch

from bc.checkpoint import LEGACY_ARCH, is_legacy_checkpoint, load_bc_model
from bc.model import BCModel
from bc.model_config import MODEL_CONFIG_DEFAULTS, build_model_cfg


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
