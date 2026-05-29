"""Unit test for the legacy-resume LR ramp (`bc.resume_warmup.WarmupSchedule`)."""

from __future__ import annotations

from bc.resume_warmup import WarmupSchedule
import pytest
import torch


def _optim(lr: float = 0.0) -> torch.optim.Optimizer:
    # A real optimizer with a throwaway param — `step` only touches param_groups.
    return torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=lr)


def test_warmup_ramps_then_holds() -> None:
    target, n = 1e-3, 4
    warmup = WarmupSchedule(target, n)
    optim = _optim()

    # Ramp: batch t (0-indexed) sets lr = target·(t+1)/N.
    for t in range(n):
        warmup.step(optim)
        assert optim.param_groups[0]["lr"] == pytest.approx(target * (t + 1) / n)
        assert warmup.t == t + 1

    # Hold: every step past N pins lr at target; counter keeps climbing.
    for extra in range(3):
        warmup.step(optim)
        assert optim.param_groups[0]["lr"] == pytest.approx(target)
        assert warmup.t == n + extra + 1


def test_warmup_sets_all_param_groups() -> None:
    target = 5e-4
    warmup = WarmupSchedule(target, n_batches=2)
    p1, p2 = torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True)
    optim = torch.optim.AdamW([{"params": [p1]}, {"params": [p2]}], lr=0.0)

    warmup.step(optim)  # first batch → target·1/2
    assert [g["lr"] for g in optim.param_groups] == pytest.approx([target / 2, target / 2])
