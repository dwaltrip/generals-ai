"""Smoke test: the BC checkpoint loads and produces the expected head shapes.

The live single-tick inference path is exercised end-to-end by `test_driver`
(full game loop) and the `test_parity` oracle in eval-tools.
"""

from pathlib import Path

import pytest
import torch

from training.bc.checkpoint import load_bc_model
from training.bc.constants import H_PADDED, W_PADDED
from training.bc.inference import default_device
from training.bc.obs_config import OBS_CHANNELS


# packages/self-play/tests/ → repo root is 3 levels up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKPOINT = _REPO_ROOT / "training" / "data" / "runs-cloud" \
    / "2026-05-23T23-40-14Z" / "checkpoints" / "epoch_005.pt"


def test_load_checkpoint_smoke():
    if not _CHECKPOINT.exists():
        pytest.skip(f"checkpoint not present at {_CHECKPOINT}")
    device = default_device()
    model = load_bc_model(_CHECKPOINT, device)
    assert isinstance(model, torch.nn.Module)

    # Forward a zero obs to confirm the loaded weights produce the expected
    # policy/pass/value head shapes.
    obs = torch.zeros((1, OBS_CHANNELS, H_PADDED, W_PADDED), device=device)
    valid_mask = torch.ones((1, 1, H_PADDED, W_PADDED), dtype=torch.bool, device=device)
    with torch.no_grad():
        out = model(obs, valid_mask)
    assert out["policy_logits"].shape == (1, 8, H_PADDED, W_PADDED)
    assert out["pass_logit"].shape == (1,)
    assert out["value_logits"].shape == (1, 8)
