import pytest
import torch

from settings import RUNS_CLOUD_DIR
from training.bc.constants import H_PADDED, W_PADDED
from training.bc.inference import default_device
from training.bc.storage.checkpoint import load_checkpoint


# TODO: Pull this from some centralized settings / configs / constants location
NUM_PLAYERS = 8

_CHECKPOINT = RUNS_CLOUD_DIR / "2026-05-23T23-40-14Z" / "checkpoints" / "epoch_005.pt"


# TODO: Why are we testing checkpoint loading here in self-play?
# I don't see anything related to self-play here.
# The file is called "test agent" but that seems like a complete misnomer.
def test_load_checkpoint_smoke():
    """Smoke test: BC checkpoint loads and produces the expected head shapes"""
    if not _CHECKPOINT.exists():
        pytest.skip(f"checkpoint not present at {_CHECKPOINT}")
    device = default_device()
    model = load_checkpoint(_CHECKPOINT, device).model
    assert isinstance(model, torch.nn.Module)

    # Use `cfg.in_chg` from the checkpoint itself to size the obs tensor.
    obs = torch.zeros((1, model.cfg.in_ch, H_PADDED, W_PADDED), device=device)
    valid_mask = torch.ones((1, 1, H_PADDED, W_PADDED), dtype=torch.bool, device=device)
    with torch.no_grad():
        out = model(obs, valid_mask)
    assert out.policy_logits.shape == (1, NUM_PLAYERS, H_PADDED, W_PADDED)
    assert out.pass_logit.shape == (1,)
    assert out.value_logits.shape == (1, NUM_PLAYERS)
