"""Per-cell directional policy head — NCHW `[B, 8, H, W]` logits.

Both halves of the action-layout pin live in this module: `PolicyHead`
produces the NESW/`sub = dir*2 + split` channel order, and
`flatten_policy_logits` is the consumer-side transform to the cell-major
flat layout the action target uses. A change to the channel order touches
both, in one file.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from training.bc.model.trunk import PyramidModule


class PolicyHead(nn.Module):
    """
    Per-cell directional policy head.

    Structure: small Pyramid Module (N=1, M=0 per design doc §4) → 3×3 conv
    projecting to 8 output channels = 4 directions × 2 splits.

    Output stays in NCHW `[B, 8, H, W]`. The caller (loss code at training
    time, inference code at eval time) does the layout transform
    `permute(0,2,3,1).reshape(B, -1)` to land at the cell-major flat layout
    that the action target uses (see `bc/actions.py` flat-layout pin and
    the cross-cutting permute contract in 5.18-3 session note).

    NESW direction order matches `bc.actions.{N,E,S,W}` and the
    sub-channel pin `sub = dir*2 + split`. Output channel 0 = N split=0,
    channel 1 = N split=1, channel 2 = E split=0, ... channel 7 = W split=1.
    """

    def __init__(self, in_ch: int):
        super().__init__()
        # Head Pyramid Module at uniform width. With M=0 at middle and
        # inner levels, the head still has its strided down/up structure
        # (no ResBlocks in between, just the strided pair at each level).
        self.head_pm = PyramidModule(
            in_ch=in_ch,
            n_outer=1, m_middle=0, m_inner=0,
            widths=(in_ch, in_ch, in_ch),
        )
        self.final_conv = nn.Conv2d(in_ch, 8, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → [B, 8, H, W]."""
        x = self.head_pm(x)
        x = self.final_conv(x)
        return x


# Mask logits set to this value before the softmax. Large-negative
# rather than `-inf` to avoid NaN propagation if a downstream op
# touches a masked position. Magnitude is bounded by FP16's representable
# range (max abs ~65504); -1e4 is comfortably inside that and still
# underflows to ~0 after softmax (`e^{-1e4} ≈ 0` in any precision).
# Earlier versions used -1e9 which silently overflowed under AMP.
MASK_NEG = -1e4


def flatten_policy_logits(
    policy_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Flatten NCHW policy logits to cell-major `[B, H·W·8]` and apply
    the legality mask. Owner of the cell-major permute contract (5.18-3
    session note); callers use the flat layout for CE / argmax.
    """
    B = policy_logits.shape[0]
    # Permute moves the direction channel last, matching the action
    # target's cell-major layout (flat_idx = cell_padded*8 + sub,
    # sub = dir*2 + split):
    #   [B, 8, H, W]  →  [B, H, W, 8]  →  [B, H·W·8]
    flat = policy_logits.permute(0, 2, 3, 1).contiguous().reshape(B, -1)
    # Apply legality mask. Illegal positions get MASK_NEG (large-negative,
    # mixed-precision-safe vs -inf) — post-softmax, prob ≈ 0.
    return flat.masked_fill(~mask.reshape(B, -1), MASK_NEG)
