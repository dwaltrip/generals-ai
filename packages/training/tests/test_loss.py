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
from training.bc.aux_heads import REGISTRY, AuxHeadSpec
from training.bc.loss import LossAccumulator, LossConfig, bc_loss
from training.bc.model import ModelOut


# The active-aux-head spec lists a caller hands to `LossAccumulator` / bakes into
# a `ModelOut` (the model's `active_aux_specs`). One head per run today (mutually
# exclusive).
TIME_BIN = (REGISTRY["time_bin"],)
NEXT_DEATH = (REGISTRY["next_death"],)


def _model_out(
    d: dict[str, torch.Tensor], specs: tuple[AuxHeadSpec, ...] = ()
) -> ModelOut:
    """Wrap a forward-output fixture dict as a `ModelOut`.
    `specs` are the active aux heads carried on the output."""
    core = ("policy_logits", "pass_logit", "value_logits")
    return ModelOut(
        policy_logits=d["policy_logits"],
        pass_logit=d["pass_logit"],
        value_logits=d["value_logits"],
        aux={k: v for k, v in d.items() if k not in core},
        active_aux_specs=specs,
    )


def _fake_losses(
    policy: float, value: float, pass_: float, n_non_pass: int,
    value_soft: float | None = None,
    elim: float | None = None, elim_soft: float | None = None,
    n_elim: int | None = None,
    next_elim: float | None = None, next_elim_soft: float | None = None,
    n_next_elim: int | None = None,
) -> dict[str, torch.Tensor]:
    """Mimic the `bc_loss` return dict shape with synthetic scalars.
    `value_soft` defaults to `value` (the τ=0 relationship). The elim /
    next_elim keys are omitted unless their hard metric is given — matching a
    non-elim `bc_loss` return (the two are mutually exclusive in practice)."""
    out = {
        "policy": torch.tensor(policy),
        "value": torch.tensor(value),
        "value_soft": torch.tensor(value if value_soft is None else value_soft),
        "pass": torch.tensor(pass_),
        "n_non_pass": torch.tensor(n_non_pass),
    }
    if elim is not None:
        out["elim"] = torch.tensor(elim)
        out["elim_soft"] = torch.tensor(elim if elim_soft is None else elim_soft)
        out["n_elim"] = torch.tensor(n_elim if n_elim is not None else 0)
    if next_elim is not None:
        out["next_elim"] = torch.tensor(next_elim)
        out["next_elim_soft"] = torch.tensor(
            next_elim if next_elim_soft is None else next_elim_soft
        )
        out["n_next_elim"] = torch.tensor(n_next_elim if n_next_elim is not None else 0)
    return out


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


def _elim_batch(B: int, n_bins: int, seed: int = 0):
    """A minimal `(model_out, targets)` pair carrying the elim head's logits +
    per-player targets, plus the other heads' tensors so `bc_loss` runs whole."""
    torch.manual_seed(seed)
    model_out = {
        "policy_logits": torch.randn(B, 8, 4, 4),
        "pass_logit": torch.randn(B),
        "value_logits": torch.randn(B, 8),
        "elim_logits": torch.randn(B, 8, n_bins),
    }
    alive = torch.ones(B, 8, dtype=torch.bool)
    alive[:, 5:] = False               # mask out a few channels (dead/phantom)
    targets = {
        "mask": torch.ones(B, 4, 4, 8, dtype=torch.bool),
        "action_target": torch.full((B,), _PASS_FLAT_IDX, dtype=torch.int64),
        "is_pass": torch.ones(B, dtype=torch.bool),
        "value_target": torch.zeros(B, dtype=torch.int64),
        "elim_bin_target": torch.randint(0, n_bins, (B, 8)),
        "alive_mask": alive,
    }
    return _model_out(model_out, TIME_BIN), targets


def test_bc_loss_elim_masked_mean_and_total_identity() -> None:
    """Elim term: `elim` is the per-player CE averaged over alive (player,
    frame) pairs only (masked-out channels don't contribute), `n_elim` is the
    alive count, and `total` reconciles with all four components including the
    elim term. τ=0 keeps `elim_soft == elim`."""
    B, n_bins = 4, 8
    model_out, targets = _elim_batch(B, n_bins, seed=1)
    cfg = LossConfig(lambda_elim=0.3)
    losses = bc_loss(model_out, targets, cfg)

    assert {"elim", "elim_soft", "n_elim"} <= losses.keys()
    alive = targets["alive_mask"]
    assert int(losses["n_elim"]) == int(alive.sum())

    # Hand-compute the masked per-player mean.
    ce = torch.nn.functional.cross_entropy(
        model_out.aux["elim_logits"].reshape(-1, n_bins),
        targets["elim_bin_target"].reshape(-1),
        reduction="none",
    ).reshape(B, 8)
    expected = (ce * alive).sum() / alive.sum()
    assert losses["elim"].item() == pytest.approx(expected.item(), rel=1e-6)
    # τ=0 → soft equals hard.
    assert losses["elim_soft"].item() == pytest.approx(losses["elim"].item(), rel=1e-6)

    # Total carries the elim term alongside the other three.
    expected_total = (
        losses["policy"]
        + cfg.lambda_value * losses["value_soft"]
        + cfg.mu_pass * losses["pass"]
        + cfg.lambda_elim * losses["elim_soft"]
    )
    assert losses["total"].item() == pytest.approx(expected_total.item(), rel=1e-6)


def test_bc_loss_elim_mask_excludes_dead_channels() -> None:
    """Masked-out channels are genuinely excluded: flipping a masked channel's
    target (which would change an *unmasked* mean) leaves `elim` unchanged."""
    B, n_bins = 3, 8
    model_out, targets = _elim_batch(B, n_bins, seed=2)
    base = bc_loss(model_out, targets, LossConfig(lambda_elim=0.5))

    # Perturb a masked (dead) channel's target — must not move the loss.
    perturbed = dict(targets)
    bt = targets["elim_bin_target"].clone()
    bt[:, 7] = (bt[:, 7] + 3) % n_bins        # channel 7 is masked off
    perturbed["elim_bin_target"] = bt
    after = bc_loss(model_out, perturbed, LossConfig(lambda_elim=0.5))
    assert after["elim"].item() == pytest.approx(base["elim"].item(), rel=1e-6)


def test_bc_loss_elim_soft_differs_at_tau() -> None:
    """τ>0 makes the soft elim CE diverge from the hard reporting CE, and the
    total reconciles against the soft objective."""
    B, n_bins = 4, 8
    model_out, targets = _elim_batch(B, n_bins, seed=3)
    cfg = LossConfig(lambda_elim=0.2, elim_target_tau=1.0)
    losses = bc_loss(model_out, targets, cfg)
    assert losses["elim_soft"].item() != pytest.approx(losses["elim"].item())
    expected_total = (
        losses["policy"]
        + cfg.lambda_value * losses["value_soft"]
        + cfg.mu_pass * losses["pass"]
        + cfg.lambda_elim * losses["elim_soft"]
    )
    assert losses["total"].item() == pytest.approx(expected_total.item(), rel=1e-6)


def test_bc_loss_no_elim_keys_without_head() -> None:
    """With no active aux specs, no elim keys appear and the total is untouched by
    `lambda_elim` — the disabled-head no-op. Gating is the `active_aux_specs` list
    (empty here), not the presence of `elim_logits`."""
    B = 4
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
        "value_target": torch.zeros(B, dtype=torch.int64),
    }
    model_out = _model_out(model_out)
    with_lambda = bc_loss(model_out, targets, LossConfig(lambda_elim=0.9))
    assert "elim" not in with_lambda and "n_elim" not in with_lambda
    baseline = bc_loss(model_out, targets, LossConfig(lambda_elim=0.0))
    assert with_lambda["total"].item() == pytest.approx(baseline["total"].item())


def test_model_out_asserts_active_head_missing_logits() -> None:
    """`ModelOut` constructor validates that the passed specs match the logits dict"""
    B = 4
    torch.manual_seed(0)
    # no `elim_logits`, but time_bin declared as an active aux head
    model_out = {
        "policy_logits": torch.randn(B, 8, 4, 4),
        "pass_logit": torch.randn(B),
        "value_logits": torch.randn(B, 8),
    }
    with pytest.raises(AssertionError, match="elim_logits"):
        _model_out(model_out, TIME_BIN)


def test_accumulator_elim_weighted_by_alive_count() -> None:
    """The accumulator weights elim means by the per-batch alive count `n_elim`
    (distinct from n_non_pass / n_samples), and the epoch total carries the
    elim term."""
    cfg = LossConfig(lambda_elim=0.4)
    acc = LossAccumulator(cfg, TIME_BIN)
    acc.update(
        _fake_losses(policy=2.0, value=1.0, pass_=0.5, n_non_pass=4,
                     elim=1.2, elim_soft=1.1, n_elim=10),
        batch_size=8,
    )
    acc.update(
        _fake_losses(policy=1.0, value=2.0, pass_=0.3, n_non_pass=2,
                     elim=0.6, elim_soft=0.5, n_elim=30),
        batch_size=8,
    )
    s = acc.summary()
    # Elim weighted by n_elim: (1.2*10 + 0.6*30) / 40, soft likewise.
    assert s["elim"] == pytest.approx((1.2 * 10 + 0.6 * 30) / 40)
    assert s["elim_soft"] == pytest.approx((1.1 * 10 + 0.5 * 30) / 40)
    assert s["n_elim"] == 40
    expected_total = (
        s["policy"] + cfg.lambda_value * s["value_soft"]
        + cfg.mu_pass * s["pass"] + cfg.lambda_elim * s["elim_soft"]
    )
    assert s["total"] == pytest.approx(expected_total)


def test_accumulator_next_death_weighted_by_victim_count() -> None:
    """The next_death analogue: the accumulator weights next_elim means by the
    per-batch defined-next-victim count `n_next_elim`, and the reconstructed epoch
    total carries the *soft* next_elim term (the core slice-2 invariant — the
    accumulator rebuilds `total` from the registry, not a hardcoded list)."""
    cfg = LossConfig(lambda_elim=0.4)
    acc = LossAccumulator(cfg, NEXT_DEATH)
    acc.update(
        _fake_losses(policy=2.0, value=1.0, pass_=0.5, n_non_pass=4,
                     next_elim=1.2, next_elim_soft=1.1, n_next_elim=10),
        batch_size=8,
    )
    acc.update(
        _fake_losses(policy=1.0, value=2.0, pass_=0.3, n_non_pass=2,
                     next_elim=0.6, next_elim_soft=0.5, n_next_elim=30),
        batch_size=8,
    )
    s = acc.summary()
    # next_elim weighted by n_next_elim: (1.2*10 + 0.6*30) / 40, soft likewise.
    assert s["next_elim"] == pytest.approx((1.2 * 10 + 0.6 * 30) / 40)
    assert s["next_elim_soft"] == pytest.approx((1.1 * 10 + 0.5 * 30) / 40)
    assert s["n_next_elim"] == 40
    # No time_bin keys leak in (that head never fired).
    assert "elim" not in s and "n_elim" not in s
    # Total reconstructs from the soft term, NOT the hard reporting CE.
    expected_total = (
        s["policy"] + cfg.lambda_value * s["value_soft"]
        + cfg.mu_pass * s["pass"] + cfg.lambda_elim * s["next_elim_soft"]
    )
    assert s["total"] == pytest.approx(expected_total)


def test_accumulator_warns_on_zero_count_active_head() -> None:
    """An active head that accumulates zero countable frames across the epoch (the
    degenerate empty case) warns loudly and skips its metrics — no NaN, no silence."""
    acc = LossAccumulator(LossConfig(lambda_elim=0.5), TIME_BIN)
    with pytest.warns(UserWarning, match="zero countable frames"):
        s = acc.summary()
    assert "elim" not in s


def test_accumulator_no_elim_keys_for_non_elim_run() -> None:
    """Without elim batches, the summary is byte-identical to before — no elim
    keys, total unchanged."""
    acc = LossAccumulator(LossConfig(lambda_elim=0.5))
    acc.update(_fake_losses(policy=2.0, value=1.0, pass_=0.5, n_non_pass=4), batch_size=8)
    s = acc.summary()
    assert "elim" not in s and "n_elim" not in s
    # lambda_elim set but no elim folded → term is 0.
    assert s["total"] == pytest.approx(
        s["policy"] + 0.5 * s["value_soft"] + 1.0 * s["pass"]
    )


def _next_death_batch(B: int, seed: int = 0):
    """A minimal `(model_out, targets)` pair for the next_death (who-is-removed-
    next) head: `[B, 8]` logits, a present mask, the next-victim target, and the
    per-channel removal horizon the soft target consumes."""
    torch.manual_seed(seed)
    model_out = {
        "policy_logits": torch.randn(B, 8, 4, 4),
        "pass_logit": torch.randn(B),
        "value_logits": torch.randn(B, 8),
        "next_elim_logits": torch.randn(B, 8),
    }
    present = torch.ones(B, 8, dtype=torch.bool)
    present[:, 5:] = False                          # removed/phantom channels
    # removal horizons: channel c is removed at ~10*c ticks out; channel 0 (self)
    # is the eventual winner (huge sentinel horizon → ~0 soft mass).
    removal_dt = (torch.arange(8, dtype=torch.int64) * 10).repeat(B, 1).clone()
    removal_dt[:, 0] = 100_000                       # winner sentinel-scale horizon
    targets = {
        "mask": torch.ones(B, 4, 4, 8, dtype=torch.bool),
        "action_target": torch.full((B,), _PASS_FLAT_IDX, dtype=torch.int64),
        "is_pass": torch.ones(B, dtype=torch.bool),
        "value_target": torch.zeros(B, dtype=torch.int64),
        "next_elim_target": torch.ones(B, dtype=torch.int64),   # victim = channel 1 (soonest real)
        "present_mask": present,
        "next_elim_removal_dt": removal_dt,
    }
    return _model_out(model_out, NEXT_DEATH), targets


def test_next_death_soft_equals_hard_at_tau_zero() -> None:
    """τ=0 keeps `next_elim_soft == next_elim` (the hard one-hot CE), and the
    total carries the soft term alongside the other heads."""
    B = 6
    model_out, targets = _next_death_batch(B, seed=1)
    cfg = LossConfig(lambda_elim=0.3)               # next_elim_target_tau defaults to 0
    losses = bc_loss(model_out, targets, cfg)
    assert {"next_elim", "next_elim_soft", "n_next_elim"} <= losses.keys()
    assert int(losses["n_next_elim"]) == B          # all frames have a defined victim
    assert losses["next_elim_soft"].item() == pytest.approx(losses["next_elim"].item())
    assert losses["total"].item() == pytest.approx(
        (losses["policy"] + cfg.lambda_value * losses["value_soft"]
         + cfg.mu_pass * losses["pass"]
         + cfg.lambda_elim * losses["next_elim_soft"]).item(), rel=1e-6
    )


def test_next_death_soft_target_shape_and_winner_mass() -> None:
    """τ>0 builds the time-based soft target `p_i ∝ exp(-(dt_i-dt_min)/τ)` over the
    present domain: it diverges from the hard CE, the winner's huge horizon gets
    ~0 mass, and the math matches a hand-rolled softmax CE."""
    B = 4
    model_out, targets = _next_death_batch(B, seed=2)
    tau = 15.0
    cfg = LossConfig(lambda_elim=0.5, next_elim_target_tau=tau)
    losses = bc_loss(model_out, targets, cfg)
    assert losses["next_elim_soft"].item() != pytest.approx(losses["next_elim"].item())

    # Reconstruct the expected soft CE independently.
    logits = model_out.aux["next_elim_logits"]
    present = targets["present_mask"]
    dt = targets["next_elim_removal_dt"].float()
    neg = torch.where(present, -dt / tau, torch.tensor(float("-inf")))
    soft_target = torch.softmax(neg, dim=1)
    # Winner channel (huge horizon) carries negligible target mass.
    assert soft_target[:, 0].max().item() < 1e-6
    logp = torch.log_softmax(logits.masked_fill(~present, float("-inf")), dim=1)
    logp = torch.where(present, logp, torch.zeros_like(logp))
    expected = -(soft_target * logp).sum(dim=1).mean()
    assert losses["next_elim_soft"].item() == pytest.approx(expected.item(), rel=1e-6)


def test_next_death_soft_target_splits_ties_evenly() -> None:
    """A removal tie (two present channels with the same horizon) gets equal soft
    mass — the property the hard one-hot's lowest-channel tie-break can't express."""
    B = 2
    model_out, targets = _next_death_batch(B, seed=4)
    # Channels 1 and 2 are removed at the same tick → equal removal_dt.
    targets["next_elim_removal_dt"][:, 1] = 30
    targets["next_elim_removal_dt"][:, 2] = 30
    tau = 15.0
    present = targets["present_mask"]
    dt = targets["next_elim_removal_dt"].float()
    neg = torch.where(present, -dt / tau, torch.tensor(float("-inf")))
    soft_target = torch.softmax(neg, dim=1)
    assert torch.allclose(soft_target[:, 1], soft_target[:, 2])
    # And the loss runs clean on the tie (no NaN).
    losses = bc_loss(
        model_out, targets, LossConfig(lambda_elim=0.3, next_elim_target_tau=tau)
    )
    assert torch.isfinite(losses["next_elim_soft"]).all()


def test_next_death_winner_tail_frames_masked() -> None:
    """Winner-tail frames (`next_elim_target == -1`) contribute nothing to either
    the hard or the soft next_elim mean, and aren't counted in `n_next_elim`."""
    B = 5
    model_out, targets = _next_death_batch(B, seed=3)
    targets["next_elim_target"][0] = -1             # frame 0 is a winner-tail frame
    cfg = LossConfig(lambda_elim=0.4, next_elim_target_tau=15.0)
    losses = bc_loss(model_out, targets, cfg)
    assert int(losses["n_next_elim"]) == B - 1      # the masked frame is excluded

    # Dropping frame 0 entirely gives the same means (it carried zero weight).
    sub_out = ModelOut(
        policy_logits=model_out.policy_logits[1:],
        pass_logit=model_out.pass_logit[1:],
        value_logits=model_out.value_logits[1:],
        aux={k: v[1:] for k, v in model_out.aux.items()},
        active_aux_specs=model_out.active_aux_specs,
    )
    sub_tgt = {k: v[1:] for k, v in targets.items()}
    sub = bc_loss(sub_out, sub_tgt, cfg)
    assert losses["next_elim"].item() == pytest.approx(sub["next_elim"].item(), rel=1e-6)
    assert losses["next_elim_soft"].item() == pytest.approx(sub["next_elim_soft"].item(), rel=1e-6)


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

    model_out = _model_out(model_out)
    losses = bc_loss(model_out, targets)

    assert int(losses["n_non_pass"]) == 0
    assert losses["policy"].item() == 0.0
    assert not torch.isnan(losses["total"]).item()

    # Gradient path stays clean: total.backward() shouldn't NaN out the
    # policy head even though policy_ce is zero.
    losses["total"].backward()
    assert model_out.policy_logits.grad is not None
    assert not torch.isnan(model_out.policy_logits.grad).any().item()


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

    model_out = _model_out(model_out)
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
