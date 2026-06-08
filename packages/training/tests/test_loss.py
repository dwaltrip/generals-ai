"""
Unit tests for `bc.loss`.

`LossAccumulator`: the load-bearing claim is the sample-weighted-mean math
— policy weights by non-pass-frame count, value/pass weight by batch size,
and total reconciles with the components via the fixed LAMBDA_VALUE /
MU_PASS weights at every aggregation level. These tests pin that math
against hand-computed values.

`bc_loss`: pins the all-pass-batch edge case, which the sync-free
implementation handles via `F.cross_entropy(reduction="sum")` returning 0
(not NaN) when every target is ignored.
"""

from __future__ import annotations

import pytest
import torch

from bc.actions import _PASS_FLAT_IDX
from bc.loss import LAMBDA_VALUE, MU_PASS, LossAccumulator, bc_loss


def _fake_losses(
    policy: float, value: float, pass_: float, n_non_pass: int
) -> dict[str, torch.Tensor]:
    """Mimic the `bc_loss` return dict shape with synthetic scalars."""
    return {
        "policy": torch.tensor(policy),
        "value": torch.tensor(value),
        "pass": torch.tensor(pass_),
        "n_non_pass": torch.tensor(n_non_pass),
    }


def test_weighted_means_and_total_reconciliation() -> None:
    acc = LossAccumulator()
    # Batch 1: 4 non-pass out of 8 frames
    acc.update(_fake_losses(policy=2.0, value=1.0, pass_=0.5, n_non_pass=4), batch_size=8)
    # Batch 2: 2 non-pass out of 8 frames
    acc.update(_fake_losses(policy=1.0, value=2.0, pass_=0.3, n_non_pass=2), batch_size=8)

    s = acc.summary()

    # Policy weighted by n_non_pass: (2.0*4 + 1.0*2) / (4 + 2) = 10/6
    assert s["policy"] == pytest.approx(10 / 6)
    # Value/pass weighted by batch_size: (X*8 + Y*8) / 16
    assert s["value"] == pytest.approx((1.0 * 8 + 2.0 * 8) / 16)
    assert s["pass"] == pytest.approx((0.5 * 8 + 0.3 * 8) / 16)
    # Total reconciles with components via the head weights.
    expected_total = s["policy"] + LAMBDA_VALUE * s["value"] + MU_PASS * s["pass"]
    assert s["total"] == pytest.approx(expected_total)

    assert s["n_non_pass"] == 6
    assert s["n_samples"] == 16


def test_all_pass_batch_contributes_zero_policy_weight() -> None:
    """An all-pass batch shouldn't bias the policy mean — n_non_pass=0 makes
    its weight zero, but value/pass still average over its samples."""
    acc = LossAccumulator()
    acc.update(_fake_losses(policy=0.0, value=1.5, pass_=0.5, n_non_pass=0), batch_size=4)
    acc.update(_fake_losses(policy=2.0, value=1.0, pass_=0.4, n_non_pass=2), batch_size=4)

    s = acc.summary()

    # Policy mean ignores the all-pass batch: (2.0 * 2) / 2 = 2.0
    assert s["policy"] == pytest.approx(2.0)
    # Value/pass include both batches: weighted by 4 each.
    assert s["value"] == pytest.approx((1.5 * 4 + 1.0 * 4) / 8)
    assert s["pass"] == pytest.approx((0.5 * 4 + 0.4 * 4) / 8)


def test_empty_accumulator_returns_zeros() -> None:
    """No updates → no div-by-zero, all fields zero."""
    s = LossAccumulator().summary()
    assert s["policy"] == 0.0
    assert s["value"] == 0.0
    assert s["pass"] == 0.0
    assert s["total"] == 0.0
    assert s["n_non_pass"] == 0
    assert s["n_samples"] == 0


def test_bc_loss_all_pass_batch_returns_zero_policy_no_nan() -> None:
    """All-pass batch must produce policy_ce = 0 (not NaN).

    The sync-free implementation removed the host-side `if n_non_pass > 0`
    guard and relies on `F.cross_entropy(reduction="sum")` returning 0 —
    not NaN — when every target is ignored. This pins that invariant so a
    future PyTorch change can't silently reintroduce NaN here.
    """
    B, H, W = 4, 8, 8

    model_out = {
        "policy_logits": torch.randn(B, 8, H, W, requires_grad=True),
        "pass_logit": torch.randn(B, requires_grad=True),
        "value_logits": torch.randn(B, 8, requires_grad=True),
    }
    targets = {
        "mask": torch.ones(B, H, W, 8, dtype=torch.bool),
        "action_target": torch.full((B,), _PASS_FLAT_IDX, dtype=torch.int64),
        "is_pass": torch.ones(B, dtype=torch.bool),
        "value_target": torch.zeros(B, dtype=torch.int64),
    }

    losses = bc_loss(model_out, targets)

    assert int(losses["n_non_pass"]) == 0
    assert losses["policy"].item() == 0.0
    assert not torch.isnan(losses["total"]).item()

    # Gradient path stays clean: total.backward() shouldn't NaN out the
    # policy head even though policy_ce is zero.
    losses["total"].backward()
    assert model_out["policy_logits"].grad is not None
    assert not torch.isnan(model_out["policy_logits"].grad).any().item()
