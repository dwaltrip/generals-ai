"""Next-elimination auxiliary head — per-player time-to-elimination logits.

Per frame and per canonical player channel (0 = perspective, 1..7 = opponents in
`opp_slots` order), predicts how soon that player is eliminated as an ordinal-
bucketed categorical. The eventual winner falls in the top "never" bin. Read off
the shared trunk to enrich it for the BC policy and the future PPO critic; not a
value-head replacement (the value head predicts the perspective's *own* final
placement, this head the *whole field's* elimination dynamics). See
`docs/2026-06/6.13-5-next-elimination-head-design.md`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EliminationHead(nn.Module):
    """Per-player elimination-bin logits pooled from the trunk embedding.

    Structure (mirrors `PassHead`'s masked global pool, widened per player·bin):

        x ─→ Conv2d(C → n_players·n_bins, k=3, pad=1)   # per-cell evidence
          ─→ × valid_mask
          ─→ masked global avg-pool over the unpadded board
          ─→ reshape [B, n_players, n_bins]

    No ReLU: we pool straight to logits, and a ReLU would discard signed
    evidence. The mask is required for the same reason as `PassHead`'s — trunk
    activations at padded cells are non-zero, and a plain pool would let the
    board-size fraction leak into the per-player magnitudes.

    The head always emits all `n_players` channels; alive-masking (real,
    currently-alive players only) is applied at loss time, not here. Pooling is
    global (not footprint-masked) for v1 — footprint/attention pooling is a
    flagged A/B.
    """

    def __init__(self, in_ch: int, n_bins: int, n_players: int = 8):
        super().__init__()
        self.n_players = n_players
        self.n_bins = n_bins
        self.conv = nn.Conv2d(in_ch, n_players * n_bins, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W], valid_mask: [B, 1, H, W] bool → [B, n_players, n_bins]."""
        m = valid_mask.to(x.dtype)
        z = self.conv(x)                                # [B, P·n_bins, H, W]
        summed = (z * m).sum(dim=(2, 3))                # [B, P·n_bins]
        count = m.sum(dim=(2, 3)).clamp(min=1.0)        # [B, 1]
        pooled = summed / count                         # [B, P·n_bins]
        return pooled.view(pooled.shape[0], self.n_players, self.n_bins)
