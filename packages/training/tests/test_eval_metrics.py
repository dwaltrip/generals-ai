"""Smoke tests for `bc.eval.metrics` — the val-pass diagnostic meters."""

import math

import torch

from training.bc.eval.metrics import ActionDistMeter, ElimMeter, PolicyEntropyMeter
from training.bc.model import MASK_NEG


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


def test_elim_meter_perfect_head_masks_and_soft_floor():
    """A near-one-hot head scores top1=1 over the alive entries (masked channels
    excluded even when their predictions are wrong), near-zero prediction
    entropy, and a τ=0 soft floor equal to the entropy of the hard target
    marginal (here uniform over 4 bins → ln 4)."""
    n_bins = 4
    bin_target = torch.tensor(
        [[0, 1, 2, 3, 0, 1, 2, 3], [3, 2, 1, 0, 3, 2, 1, 0]]
    )
    logits = torch.zeros(2, 8, n_bins)
    logits.scatter_(2, bin_target.unsqueeze(-1), 10.0)   # peak on the target bin
    alive = torch.ones(2, 8, dtype=torch.bool)
    alive[:, 6:] = False                                  # mask last two channels
    logits[:, 6:, :] = 0.0                                # corrupt them (must be ignored)

    m = ElimMeter(n_bins=n_bins, target_tau=0.0)
    m.update(logits, bin_target, alive)

    assert m.top1() == 1.0                                # masked wrong preds excluded
    ent = m.pred_entropy()
    assert ent is not None and ent < 0.01                # near one-hot
    floor = m.soft_floor()
    assert floor is not None and math.isclose(floor, math.log(4), rel_tol=1e-6)


def test_elim_meter_tau_inflates_soft_floor():
    """The softness tax is baked into the floor: with all targets in one bin the
    hard marginal entropy is 0, but a τ>0 smoothed marginal has positive
    entropy — so `elim_soft` is compared against a floor that already pays the
    same tax it does."""
    bin_target = torch.zeros(1, 8, dtype=torch.long)      # all bin 0
    logits = torch.zeros(1, 8, 4)
    alive = torch.ones(1, 8, dtype=torch.bool)

    hard = ElimMeter(4, target_tau=0.0)
    hard.update(logits, bin_target, alive)
    soft = ElimMeter(4, target_tau=1.0)
    soft.update(logits, bin_target, alive)

    hard_floor = hard.soft_floor()
    assert hard_floor is not None and hard_floor < 1e-9   # degenerate marginal ≈ 0
    assert soft.soft_floor() > 0.5                        # smoothing lifts it well clear


def test_elim_meter_empty_returns_none():
    m = ElimMeter(4, target_tau=0.0)
    assert m.top1() is None
    assert m.pred_entropy() is None
    assert m.soft_floor() is None
