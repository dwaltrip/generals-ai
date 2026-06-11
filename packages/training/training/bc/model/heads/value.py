"""Categorical placement value head — 8-way logits `[B, 8]`.

One deviation from the inherited DeepNash design lives here: the inline
value head (option B from the 5.20-2 session) — flagged in `ValueHead`,
with the full-spec extension available as the "pyramid" variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.bc.model.trunk import PyramidModule
from training.bc.model_config import VALUE_HEAD_VARIANTS


class ValueHead(nn.Module):
    """
    Categorical placement value head (8-way: 1st through 8th).

    Two architectural variants, selected by `variant`:

      "direct" (v1 spike choice — formerly "option B" in 5.20-2 / 5.20-3):

          x ─→ Conv2d(C → 1, k=3) → ReLU → mask → flatten → Linear(H·W, 8)

        Thin head — the Linear does all spatial integration. 3×3 receptive
        field per cell before flatten; ~9k params. Anomalously small relative
        to the policy head's ~1.1M.

      "pyramid" (full DeepNash spec — formerly "option C" in 5.20-2 / 5.20-3):

          x ─→ PyramidModule(N=0/M=0/M=0, uniform width=C)
            ─→ Conv2d(C → 1, k=3) → ReLU → mask → flatten → Linear(H·W, 8)

        Adds a shape-preserving U-Net-style head before proj_conv: 32→16→8
        encoder + symmetric decoder with skip connections. Each cell in the
        pre-flatten map encodes global board context via the 8×8 bottleneck,
        which is the right shape for a placement-prediction (global) task.
        ~+0.82M params; brings the value head into the same size class as
        the policy head.

    Common to both: the mask multiply between ReLU and flatten zeroes
    padded-cell contributions, so the Linear layer doesn't see per-game-
    varying junk at padded positions.

    Optional head-side dropout (anti-memorization, train-time only — see
    `ModelConfig` field docs): channel dropout on the post-`pre` features,
    elementwise dropout on the flattened vector before the Linear. Both
    default off (p=0 ≡ identity), so the modules are unconditional.
    """

    def __init__(
        self,
        in_ch: int,
        H: int,
        W: int,
        n_classes: int = 8,
        variant: str = "direct",
        dropout2d_p: float = 0.0,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        if variant not in VALUE_HEAD_VARIANTS:
            raise ValueError(
                f"variant must be one of {VALUE_HEAD_VARIANTS}; got {variant!r}"
            )
        self.variant = variant
        if variant == "pyramid":
            self.pre = PyramidModule(
                in_ch=in_ch, n_outer=0, m_middle=0, m_inner=0,
                widths=(in_ch, in_ch, in_ch),
            )
        else:
            self.pre = nn.Identity()
        self.dropout2d = nn.Dropout2d(dropout2d_p)
        self.proj_conv = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(H * W, n_classes)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W], valid_mask: [B, 1, H, W] bool → [B, n_classes]."""
        x = self.pre(x)                     # [B, C, H, W] (PM or Identity)
        x = self.dropout2d(x)
        x = self.proj_conv(x)               # [B, 1, H, W]
        x = F.relu(x)
        x = x * valid_mask.to(x.dtype)      # zero padded contributions
        x = x.flatten(1)                    # [B, H·W]
        x = self.dropout(x)
        x = self.linear(x)                  # [B, n_classes]
        return x
