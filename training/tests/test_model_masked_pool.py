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

import torch

from bc.constants import H_PADDED, W_PADDED
from bc.model import PassHead, ValueHead


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
