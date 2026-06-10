"""
Unit tests for `bc.loss`.

`LossAccumulator`: the load-bearing claim is the sample-weighted-mean math
— policy weights by non-pass-frame count, value/pass weight by batch size,
and total reconciles with the components via the `LossConfig` weights at
every aggregation level. These tests pin that math against hand-computed
values.

`bc_loss`: pins the all-pass-batch edge case, which the sync-free
implementation handles via `F.cross_entropy(reduction="sum")` returning 0
(not NaN) when every target is ignored; plus the soft-ordinal-target path
(τ=0 ≡ one-hot, τ>0 kernel shape + total uses the soft CE).
"""

from __future__ import annotations

import math

import pytest
import torch

from training.bc.actions import _PASS_FLAT_IDX
from training.bc.loss import LossAccumulator, LossConfig, bc_loss


def _fake_losses(
    policy: float, value: float, pass_: float, n_non_pass: int,
    value_soft: float | None = None,
) -> dict[str, torch.Tensor]:
    """Mimic the `bc_loss` return dict shape with synthetic scalars.
    `value_soft` defaults to `value` (the τ=0 relationship)."""
    return {
        "policy": torch.tensor(policy),
        "value": torch.tensor(value),
        "value_soft": torch.tensor(value if value_soft is None else value_soft),
        "pass": torch.tensor(pass_),
        "n_non_pass": torch.tensor(n_non_pass),
    }


def test_weighted_means_and_total_reconciliation() -> None:
    cfg = LossConfig()
    acc = LossAccumulator(cfg)
    # Batch 1: 4 non-pass out of 8 frames. value_soft deliberately differs
    # from value to pin that it accumulates independently and that the
    # total derives from the soft component.
    acc.update(
        _fake_losses(policy=2.0, value=1.0, pass_=0.5, n_non_pass=4, value_soft=0.8),
        batch_size=8,
    )
    # Batch 2: 2 non-pass out of 8 frames
    acc.update(
        _fake_losses(policy=1.0, value=2.0, pass_=0.3, n_non_pass=2, value_soft=1.6),
        batch_size=8,
    )

    s = acc.summary()

    # Policy weighted by n_non_pass: (2.0*4 + 1.0*2) / (4 + 2) = 10/6
    assert s["policy"] == pytest.approx(10 / 6)
    # Value/pass weighted by batch_size: (X*8 + Y*8) / 16
    assert s["value"] == pytest.approx((1.0 * 8 + 2.0 * 8) / 16)
    assert s["value_soft"] == pytest.approx((0.8 * 8 + 1.6 * 8) / 16)
    assert s["pass"] == pytest.approx((0.5 * 8 + 0.3 * 8) / 16)
    # Total reconciles with components via the cfg weights — and uses the
    # soft value CE (the trained objective), not the hard reporting metric.
    expected_total = (
        s["policy"] + cfg.lambda_value * s["value_soft"] + cfg.mu_pass * s["pass"]
    )
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


def test_soft_value_targets() -> None:
    """Soft ordinal placement targets: τ=0 must reproduce one-hot CE exactly
    (the inert-default guarantee every baseline run relies on), and τ>0 must
    produce a valid spread distribution whose CE drives the total."""
    B = 6
    torch.manual_seed(0)
    model_out = {
        "policy_logits": torch.randn(B, 8, 4, 4),
        "pass_logit": torch.randn(B),
        "value_logits": torch.randn(B, 8),
    }
    targets = {
        "mask": torch.ones(B, 4, 4, 8, dtype=torch.bool),
        "action_target": torch.full((B,), _PASS_FLAT_IDX, dtype=torch.int64),
        "is_pass": torch.ones(B, dtype=torch.bool),
        # Include the edge ranks (0 and 7) — their kernel rows are the
        # one-sided-neighborhood case the row softmax renormalizes.
        "value_target": torch.tensor([0, 1, 3, 5, 7, 7], dtype=torch.int64),
    }

    # τ=0: value_soft IS the hard CE (same tensor), total identical to default.
    default = bc_loss(model_out, targets)
    assert default["value_soft"] is default["value"]

    # τ>0: soft CE differs from hard; total reconciles against it.
    tau = 0.6
    cfg = LossConfig(value_target_tau=tau)
    soft = bc_loss(model_out, targets, cfg)
    assert soft["value"] == pytest.approx(default["value"].item())  # hard CE unchanged
    assert soft["value_soft"].item() != pytest.approx(soft["value"].item())
    expected_total = (
        soft["policy"] + cfg.lambda_value * soft["value_soft"] + cfg.mu_pass * soft["pass"]
    )
    assert soft["total"].item() == pytest.approx(expected_total.item())

    # Kernel shape: rows are distributions; for an interior rank, the
    # adjacent-rank mass relative to the peak is exp(-1/τ) (the documented
    # τ-picking relationship).
    from training.bc.loss import _soft_target_kernel

    kernel = _soft_target_kernel(tau, 8, torch.device("cpu"))
    assert torch.allclose(kernel.sum(dim=1), torch.ones(8))
    row = kernel[3]
    adjacent_ratio = math.exp(-1 / tau)
    assert row.argmax().item() == 3
    assert (row[4] / row[3]).item() == pytest.approx(adjacent_ratio, rel=1e-5)
    assert (row[2] / row[3]).item() == pytest.approx(adjacent_ratio, rel=1e-5)  # symmetric
