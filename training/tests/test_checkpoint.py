"""`load_bc_model` round-trips the legacy (bare state_dict) checkpoint format,
plus the combined-vs-legacy format detection used by the resume path."""

from __future__ import annotations

import torch

from bc.checkpoint import is_legacy_checkpoint, load_bc_model
from bc.model import BCModel


def test_load_bc_model_bare_state_dict(tmp_path):
    device = torch.device("cpu")
    src = BCModel(value_head_variant="direct")
    ckpt = tmp_path / "epoch_001.pt"
    # Bare state_dict is the format `_save_checkpoint` writes today.
    torch.save(src.state_dict(), ckpt)

    loaded = load_bc_model(ckpt, device)

    assert isinstance(loaded, BCModel)
    assert not loaded.training  # returned in eval mode
    src_sd, loaded_sd = src.state_dict(), loaded.state_dict()
    assert src_sd.keys() == loaded_sd.keys()
    for k in src_sd:
        assert torch.equal(src_sd[k], loaded_sd[k].cpu()), f"param mismatch: {k}"


def test_is_legacy_checkpoint_detection(tmp_path):
    legacy = tmp_path / "legacy.pt"
    combined = tmp_path / "combined.pt"
    torch.save(BCModel(value_head_variant="direct").state_dict(), legacy)  # bare state_dict
    torch.save({"model": {}, "optim": {}, "epoch": 3}, combined)           # combined wrapper

    assert is_legacy_checkpoint(legacy) is True
    assert is_legacy_checkpoint(combined) is False
