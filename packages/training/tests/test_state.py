"""TrainingState.save / from_checkpoint round-trips model + optim + epoch."""

from __future__ import annotations

import torch

from training.bc.checkpoint import LEGACY_ARCH
from training.bc.constants import H_PADDED, W_PADDED
from training.bc.model import BCModel
from training.bc.obs_config import OBS_CHANNELS
from training.bc.state import TrainingState
from training.bc.train_config import TrainConfig


def _config(tmp_path):
    # Paths are unused by fresh()/save()/from_checkpoint() and not existence-
    # checked at construction, so dummy values are fine. Default arch (direct).
    return TrainConfig(
        manifest=tmp_path / "m.json",
        intermediate=tmp_path / "i",
        run_dir=tmp_path / "run",
    )


def _one_step(state: TrainingState, device: torch.device) -> None:
    """One synthetic optimizer step so the optim accumulates moment buffers."""
    obs = torch.zeros(2, OBS_CHANNELS, H_PADDED, W_PADDED, device=device)
    valid_mask = torch.ones(2, 1, H_PADDED, W_PADDED, dtype=torch.bool, device=device)
    out = state.model(obs, valid_mask)
    loss = out.policy_logits.sum() + out.pass_logit.sum() + out.value_logits.sum()
    state.optim.zero_grad()
    state.scaler.scale(loss).backward()
    state.scaler.step(state.optim)
    state.scaler.update()


def test_training_state_round_trip(tmp_path):
    device = torch.device("cpu")
    config = _config(tmp_path)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    src = TrainingState.fresh(config, device, "test-sha")
    _one_step(src, device)  # non-trivial optim state before saving
    src.epoch = 3
    path = src.save(ckpt_dir)

    # On-disk v1 format — self-describing via the `config` block.
    raw = torch.load(path, weights_only=True)
    assert set(raw) == {"model", "optim", "scaler", "epoch", "in_ch", "config", "code_sha"}
    assert raw["epoch"] == 3
    assert raw["code_sha"] == "test-sha"
    assert path.name == "epoch_003.pt"

    restored = TrainingState.from_checkpoint(path, config, device, "test-sha")
    assert restored.epoch == 3
    assert restored.model.cfg == src.model.cfg  # arch round-trips

    # Model weights round-trip exactly.
    s, r = src.model.state_dict(), restored.model.state_dict()
    assert s.keys() == r.keys()
    for k in s:
        assert torch.equal(s[k], r[k]), f"model param mismatch: {k}"

    # AdamW moment buffers round-trip — the optimizer trajectory is preserved.
    src_opt = src.optim.state_dict()["state"]
    res_opt = restored.optim.state_dict()["state"]
    assert src_opt.keys() == res_opt.keys()
    for pid in src_opt:
        for buf in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(src_opt[pid][buf], res_opt[pid][buf]), f"optim {buf} @ {pid}"


def test_from_checkpoint_legacy_uses_fallback_epoch(tmp_path):
    """A legacy bare-state_dict has no epoch — resume passes the filename-derived
    parent epoch as the fallback, and the warmup field is left for the resume
    path to attach (not set here)."""
    device = torch.device("cpu")
    config = _config(tmp_path)
    ckpt = tmp_path / "epoch_007.pt"
    # Legacy-shaped bare state_dict (the LEGACY_ARCH fallback reconstructs it).
    torch.save(BCModel(LEGACY_ARCH).state_dict(), ckpt)

    restored = TrainingState.from_checkpoint(ckpt, config, device, "test-sha", fallback_epoch=7)
    assert restored.epoch == 7      # from the filename, not the 0 default
    assert restored.warmup is None


def test_from_checkpoint_combined_ignores_fallback_epoch(tmp_path):
    """A combined checkpoint carries its own epoch — fallback is inert."""
    device = torch.device("cpu")
    config = _config(tmp_path)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    src = TrainingState.fresh(config, device, "test-sha")
    src.epoch = 4
    path = src.save(ckpt_dir)

    restored = TrainingState.from_checkpoint(path, config, device, "test-sha", fallback_epoch=99)
    assert restored.epoch == 4      # the checkpoint's own epoch wins
