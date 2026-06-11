"""Scalar pass head — one pre-sigmoid logit per sample, masked-pooled.

(Filename carries the `_head` suffix because `pass` is a Python keyword and
can't be a module name.)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PassHead(nn.Module):
    """
    Scalar pass logit pooled from the trunk embedding.

    Structure: masked global avg pool over the unpadded region of the
    [B, C, H, W] trunk output → Linear(C, 1) → [B] scalar logit per sample.
    Loss code applies `binary_cross_entropy_with_logits` against `is_pass`.

    The mask is required because trunk activations at padded positions are
    non-zero (convs see the zero-padded input and produce output for the
    padded region). A plain AdaptiveAvgPool2d would average those into the
    pool and the per-sample dilution would scale with how much of the 32×32
    grid is real board — a board-size leak into the pass signal.
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.linear = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W], valid_mask: [B, 1, H, W] bool → [B] pre-sigmoid logit."""
        m = valid_mask.to(x.dtype)
        summed = (x * m).sum(dim=(2, 3))                # [B, C]
        count = m.sum(dim=(2, 3)).clamp(min=1.0)        # [B, 1]
        pooled = summed / count                         # [B, C]
        return self.linear(pooled).squeeze(-1)          # [B]
