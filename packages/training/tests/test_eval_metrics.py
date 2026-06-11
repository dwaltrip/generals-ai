"""Smoke tests for `bc.eval.metrics` — the val-pass diagnostic meters."""

import math

import torch

from training.bc.eval.metrics import PolicyEntropyMeter
from training.bc.loss import MASK_NEG


def _uniform_masked_logits(n_frames: int, n_legal: int, n_total: int = 32) -> torch.Tensor:
    """Flat masked logits with a uniform distribution over `n_legal` positions
    (logit 0) and the rest MASK_NEG-filled — entropy is exactly ln(n_legal)."""
    logits = torch.full((n_frames, n_total), MASK_NEG)
    logits[:, :n_legal] = 0.0
    return logits


def test_uniform_entropy_and_non_pass_weighting():
    meter = PolicyEntropyMeter()
    # 3 non-pass frames, uniform over 4 legal actions → H = ln 4 each.
    meter.update(_uniform_masked_logits(3, 4), non_pass=torch.tensor([True] * 3))
    # Uniform over 8, but only 1 of the 2 frames is non-pass.
    meter.update(_uniform_masked_logits(2, 8), non_pass=torch.tensor([True, False]))
    expected = (3 * math.log(4) + 1 * math.log(8)) / 4
    mean = meter.mean()
    assert mean is not None
    assert math.isclose(mean, expected, rel_tol=1e-5)


def test_empty_meter_returns_none():
    assert PolicyEntropyMeter().mean() is None
