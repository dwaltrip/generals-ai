"""Smoke tests for `bc.eval.metrics` — the val-pass diagnostic meters."""

import math

import torch

from training.bc.eval.metrics import ActionDistMeter, PolicyEntropyMeter
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


def test_action_dist_meter_buckets_and_pass_exclusion():
    meter = ActionDistMeter()
    # flat % 8 picks the sub-action bucket: 16→0 (n_move), 9→1 (n_split),
    # 3→3 (e_split). Row 2 is a pass frame (target -1) and must not count.
    top1 = torch.tensor([16, 9, 5, 3])
    target = torch.tensor([8, 9, -1, 11])
    non_pass = torch.tensor([True, True, False, True])
    meter.update(top1, target, non_pass)

    pred = meter.pred_dist()
    assert pred["n_move"] == pred["n_split"] == pred["e_split"] == 1 / 3
    assert pred["s_split"] == 0.0  # row 2's 5→s_split excluded with its frame
    target_d = meter.target_dist()
    assert target_d["n_move"] == target_d["n_split"] == target_d["e_split"] == 1 / 3


def test_action_dist_meter_empty_returns_none_dicts():
    meter = ActionDistMeter()
    assert all(v is None for v in meter.pred_dist().values())
    assert all(v is None for v in meter.target_dist().values())
