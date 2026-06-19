"""
Smoke tests for the valid-mask plumbing through PassHead and ValueHead.

Both heads take a per-sample `valid_mask: [B, 1, H, W] bool` indicating the
unpadded board region. PassHead uses it as a masked global-avg-pool; ValueHead
uses it to zero padded contributions before the flatten → linear projection.

These tests pin:
  (1) Full-board case: PassHead masked pool reduces to plain AdaptiveAvgPool2d.
  (2) Partial board: PassHead pool ignores padded cells (the math the fix
      exists for).
  (3) ValueHead is invariant to junk at padded positions — adding noise
      outside the unpadded region doesn't change the head's output.
"""

from __future__ import annotations

import pytest
import torch

from training.bc.constants import H_PADDED, W_PADDED
from training.bc.model import (
    BCModel,
    ElimNextDeathHead,
    ElimTimeBinHead,
    PassHead,
    ValueHead,
)
from training.bc.model_config import build_model_cfg
from training.bc.obs_config import OBS_CHANNELS


def test_pass_head_full_board_matches_plain_global_pool() -> None:
    """Full-board mask: masked pool == plain AdaptiveAvgPool2d (sanity)."""
    torch.manual_seed(0)
    in_ch = 16
    head = PassHead(in_ch=in_ch)
    head.eval()

    x = torch.randn(2, in_ch, H_PADDED, W_PADDED)
    valid_mask = torch.ones(2, 1, H_PADDED, W_PADDED, dtype=torch.bool)

    out_masked = head(x, valid_mask)

    # Reference: plain GAP → same linear layer.
    pooled = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
    out_plain = head.linear(pooled).squeeze(-1)

    assert torch.allclose(out_masked, out_plain, atol=1e-6)


def test_pass_head_partial_board_ignores_padded_cells() -> None:
    """Padded cells should not influence the pool. Stuff junk into the
    padded region and confirm the head output matches a clean run that
    only has real-board activations."""
    torch.manual_seed(1)
    in_ch = 16
    H, W = 20, 18   # unpadded; arbitrary < H_PADDED, W_PADDED
    head = PassHead(in_ch=in_ch)
    head.eval()

    # Clean input: zero everywhere outside [:H, :W]
    x_clean = torch.zeros(1, in_ch, H_PADDED, W_PADDED)
    x_clean[0, :, :H, :W] = torch.randn(in_ch, H, W)

    # Same real-board activations, junk in the padded region
    x_dirty = x_clean.clone()
    x_dirty[0, :, H:, :] = torch.randn(in_ch, H_PADDED - H, W_PADDED) * 100.0
    x_dirty[0, :, :, W:] = torch.randn(in_ch, H_PADDED, W_PADDED - W) * 100.0

    valid_mask = torch.zeros(1, 1, H_PADDED, W_PADDED, dtype=torch.bool)
    valid_mask[0, 0, :H, :W] = True

    out_clean = head(x_clean, valid_mask)
    out_dirty = head(x_dirty, valid_mask)

    assert torch.allclose(out_clean, out_dirty, atol=1e-5)


def test_value_head_variant_direct_is_default_and_baseline() -> None:
    """Default variant 'direct' constructs without the pre-projection PM
    and matches the historical param count."""
    m = BCModel()
    assert m.value_head.variant == "direct"
    # The 'direct' variant's pre-projection stage is an Identity (no params).
    assert sum(p.numel() for p in m.value_head.pre.parameters()) == 0
    # Param count = the post-mask-fix baseline (4_185_442 at 96 obs channels)
    # plus the player-status channels' first-conv expansion (128*14*9 = 16_128).
    assert sum(p.numel() for p in m.parameters()) == 4_201_570


def test_value_head_variant_pyramid_adds_expected_capacity() -> None:
    """'pyramid' variant constructs, runs forward, and adds ~0.82M params."""
    m_direct = BCModel(build_model_cfg(value_head_variant="direct"))
    m_pyr = BCModel(build_model_cfg(value_head_variant="pyramid"))
    assert m_pyr.value_head.variant == "pyramid"

    n_direct = sum(p.numel() for p in m_direct.parameters())
    n_pyr = sum(p.numel() for p in m_pyr.parameters())
    delta = n_pyr - n_direct
    # Tight bound — head_pm at uniform 128 has a known param count.
    assert 800_000 < delta < 850_000, (
        f"expected pyramid to add ~822k params; got {delta:,}"
    )

    # Forward pass shape is identical to direct (the variant only changes
    # internal capacity, not the output contract).
    m_pyr.eval()
    obs = torch.zeros(2, OBS_CHANNELS, H_PADDED, W_PADDED)
    vm = torch.ones(2, 1, H_PADDED, W_PADDED, dtype=torch.bool)
    with torch.no_grad():
        out = m_pyr(obs, vm)
    assert out["value_logits"].shape == (2, 8)
    assert out["policy_logits"].shape == (2, 8, H_PADDED, W_PADDED)
    assert out["pass_logit"].shape == (2,)


def test_elim_head_disabled_by_default_adds_no_state() -> None:
    """The arch gate: a default (disabled) model has `elim_head is None`, no
    elim state_dict keys, and emits no `elim_logits` — the backward-compat
    contract that lets every pre-elim checkpoint load strict=True."""
    m = BCModel()
    assert m.elim_head is None
    assert not any("elim" in k for k in m.state_dict())

    m.eval()
    obs = torch.zeros(2, OBS_CHANNELS, H_PADDED, W_PADDED)
    vm = torch.ones(2, 1, H_PADDED, W_PADDED, dtype=torch.bool)
    with torch.no_grad():
        out = m(obs, vm)
    assert "elim_logits" not in out


def test_elim_head_time_bin_shape_and_isolated_keys() -> None:
    """time_bin variant: forward emits `elim_logits` shaped [B, 8, n_bins], and
    the only state_dict keys added vs the off model live under `elim_head.`."""
    cfg = build_model_cfg(elim_head_variant="time_bin")
    m = BCModel(cfg)
    assert m.elim_head is not None

    m.eval()
    obs = torch.zeros(3, OBS_CHANNELS, H_PADDED, W_PADDED)
    vm = torch.ones(3, 1, H_PADDED, W_PADDED, dtype=torch.bool)
    with torch.no_grad():
        out = m(obs, vm)
    assert out["elim_logits"].shape == (3, 8, cfg.elim_n_bins)
    assert "next_elim_logits" not in out

    extra = set(m.state_dict()) - set(BCModel().state_dict())
    assert extra and all(k.startswith("elim_head.") for k in extra)


def test_elim_head_next_death_shape_and_isolated_keys() -> None:
    """next_death variant: forward emits `next_elim_logits` shaped [B, 8] (one
    per-player logit, the cross-player softmax is taken at loss time), and the
    only added state_dict keys live under `elim_head.`."""
    cfg = build_model_cfg(elim_head_variant="next_death")
    m = BCModel(cfg)
    assert m.elim_head is not None

    m.eval()
    obs = torch.zeros(3, OBS_CHANNELS, H_PADDED, W_PADDED)
    vm = torch.ones(3, 1, H_PADDED, W_PADDED, dtype=torch.bool)
    with torch.no_grad():
        out = m(obs, vm)
    assert out["next_elim_logits"].shape == (3, 8)
    assert "elim_logits" not in out

    extra = set(m.state_dict()) - set(BCModel().state_dict())
    assert extra and all(k.startswith("elim_head.") for k in extra)


def test_elim_head_masked_pool_excludes_padded_cells() -> None:
    """The masked global pool averages only over unpadded cells. Pinned against
    a hand computation (mirrors the ValueHead test) — a plain pool would dilute
    the per-player logits by the padded-board fraction. Input-invariance can't
    be tested directly: the 3×3 conv bleeds padded junk one cell into the valid
    region before the pool runs."""
    torch.manual_seed(3)
    in_ch, H, W = 16, 20, 18
    head = ElimTimeBinHead(in_ch=in_ch, n_bins=8)
    head.eval()

    x = torch.randn(1, in_ch, H_PADDED, W_PADDED)
    valid_mask = torch.zeros(1, 1, H_PADDED, W_PADDED, dtype=torch.bool)
    valid_mask[0, 0, :H, :W] = True

    out = head(x, valid_mask)

    z = head.conv(x)
    m = valid_mask.to(z.dtype)
    pooled = (z * m).sum(dim=(2, 3)) / m.sum(dim=(2, 3)).clamp(min=1.0)
    expected = pooled.view(1, 8, 8)

    assert out.shape == (1, 8, 8)
    assert torch.allclose(out, expected, atol=1e-6)


def test_elim_head_lse_pool_finite_with_finite_beta_grad() -> None:
    """The `lse` readout (masked log-sum-exp, learnable β) yields finite logits
    AND a finite β gradient on a partial board. Regression: masking z with -inf
    feeds 0·(-inf)=NaN into β's grad through logsumexp — the head masks in
    exponent-space with a finite sentinel instead."""
    torch.manual_seed(4)
    in_ch, H, W = 16, 20, 18
    head = ElimTimeBinHead(in_ch=in_ch, n_bins=8, pool="lse")

    x = torch.randn(2, in_ch, H_PADDED, W_PADDED, requires_grad=True)
    valid_mask = torch.zeros(2, 1, H_PADDED, W_PADDED, dtype=torch.bool)
    valid_mask[:, 0, :H, :W] = True

    out = head(x, valid_mask)
    assert out.shape == (2, 8, 8)
    assert torch.isfinite(out).all()

    out.pow(2).mean().backward()
    assert head.raw_beta.grad is not None and torch.isfinite(head.raw_beta.grad).all()
    assert torch.isfinite(x.grad).all()


def test_elim_head_lse_pool_lies_between_masked_mean_and_max() -> None:
    """The smooth-max lse readout sits between the masked mean and masked max
    per (player, bin) — its defining interpolation property, and a check that
    the pool excludes padded cells in both the sum and the max."""
    torch.manual_seed(5)
    in_ch, H, W = 16, 20, 18
    head = ElimTimeBinHead(in_ch=in_ch, n_bins=8, pool="lse")
    head.eval()

    x = torch.randn(1, in_ch, H_PADDED, W_PADDED)
    valid_mask = torch.zeros(1, 1, H_PADDED, W_PADDED, dtype=torch.bool)
    valid_mask[0, 0, :H, :W] = True

    out = head(x, valid_mask).view(1, -1)               # [1, 8·8] lse logits

    z = head.conv(x).flatten(2)                          # [1, 64, H·W]
    mb = valid_mask.flatten(2)                            # [1, 1, H·W]
    mean_p = (z * mb).sum(2) / mb.sum(2).clamp(min=1)
    max_p = z.masked_fill(~mb.bool(), float("-inf")).amax(2)

    tol = 1e-4
    assert (out >= mean_p - tol).all() and (out <= max_p + tol).all()
    # β≈0.5 → strictly interior, not collapsed onto either bound
    assert (out > mean_p + tol).any() and (out < max_p - tol).any()


def test_next_death_head_shape_and_finite_beta_grad() -> None:
    """The who-dies-next head pools to [B, 8] (one logit per player) via the
    masked lse readout, with finite logits and a finite β gradient on a partial
    board — the same exponent-space masking guard as the time_bin lse pool."""
    torch.manual_seed(6)
    in_ch, H, W = 16, 20, 18
    head = ElimNextDeathHead(in_ch=in_ch)

    x = torch.randn(2, in_ch, H_PADDED, W_PADDED, requires_grad=True)
    valid_mask = torch.zeros(2, 1, H_PADDED, W_PADDED, dtype=torch.bool)
    valid_mask[:, 0, :H, :W] = True

    out = head(x, valid_mask)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()

    out.pow(2).mean().backward()
    assert head.raw_beta.grad is not None and torch.isfinite(head.raw_beta.grad).all()
    assert torch.isfinite(x.grad).all()


def test_elim_head_capacity_adds_prepool_conv() -> None:
    """`hidden > 0` inserts Conv→ReLU before the readout conv: the head still
    emits [B, 8, n_bins] and gains the pre-stage's parameters."""
    thin = ElimTimeBinHead(in_ch=192, n_bins=8, hidden=0)
    wide = ElimTimeBinHead(in_ch=192, n_bins=8, hidden=192)
    assert sum(p.numel() for p in thin.pre.parameters()) == 0
    assert sum(p.numel() for p in wide.pre.parameters()) > 0

    x = torch.randn(2, 192, H_PADDED, W_PADDED)
    vm = torch.ones(2, 1, H_PADDED, W_PADDED, dtype=torch.bool)
    assert wide(x, vm).shape == (2, 8, 8)


def test_value_head_invalid_variant_raises() -> None:
    with pytest.raises(ValueError, match="variant must be one of"):
        ValueHead(in_ch=128, H=H_PADDED, W=W_PADDED, variant="nope")


def test_value_head_zeros_padded_before_flatten() -> None:
    """The mask multiply happens between ReLU and flatten — pin that
    directly. Whatever the proj_conv produces at padded positions gets
    zeroed before the linear layer sees it.

    (We can't test invariance-to-padded-junk in the input because the
    proj_conv has receptive field 3 and will leak padded junk into the
    valid-region output; the fix is specifically about the linear-layer
    exposure, not about conv-bleed.)
    """
    torch.manual_seed(2)
    in_ch = 16
    H, W = 22, 16
    head = ValueHead(in_ch=in_ch, H=H_PADDED, W=W_PADDED)
    head.eval()

    trunk_out = torch.randn(1, in_ch, H_PADDED, W_PADDED)
    valid_mask = torch.zeros(1, 1, H_PADDED, W_PADDED, dtype=torch.bool)
    valid_mask[0, 0, :H, :W] = True

    out = head(trunk_out, valid_mask)

    # Reference: do the math by hand and zero the padded region explicitly
    # between ReLU and flatten.
    x = head.proj_conv(trunk_out)
    x = torch.relu(x)
    x_zeroed = torch.zeros_like(x)
    x_zeroed[..., :H, :W] = x[..., :H, :W]
    expected = head.linear(x_zeroed.flatten(1))

    assert torch.allclose(out, expected, atol=1e-6)
