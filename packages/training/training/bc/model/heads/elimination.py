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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EliminationHead(nn.Module):
    """Per-player elimination-bin logits pooled from the trunk embedding.

    Structure (mirrors `PassHead`'s masked pool, widened per player·bin):

        x ─→ [pre: Conv2d(C → hidden, k=3) → ReLU]   # optional, `hidden > 0`
          ─→ Conv2d(· → n_players·n_bins, k=3, pad=1)  # per-cell evidence
          ─→ × valid_mask
          ─→ masked spatial pool over the unpadded board
          ─→ reshape [B, n_players, n_bins]

    Two readout knobs, both flagged A/Bs in 6.13-5, here config-gated:

    - **pool** (`"mean"` | `"lse"`). `mean` is the masked global average (v1):
      it dilutes spatially-localized elimination evidence (danger at ~3 of ~200
      tiles is a ~1.5% perturbation of the mean). `lse` is a masked, length-
      normalized log-sum-exp — a smooth maximum with a learnable temperature β
      that interpolates mean (β→0) and max (β→∞), so a single strong cell can
      surface a localized signal regardless of board size.
    - **hidden** (pre-pool capacity). `0` keeps the single linear readout conv
      (v1); `> 0` inserts `Conv(C→hidden)→ReLU` first, letting the head form a
      *nonlinear* per-cell feature (e.g. "weak own general AND strong adjacent
      enemy") the linear conv cannot. Only pays off paired with a peak-preserving
      pool — mean dilutes even a perfect per-cell detector.

    No ReLU on the readout conv: we pool straight to logits, and a ReLU would
    discard signed evidence. The mask is required for the same reason as
    `PassHead`'s — trunk activations at padded cells are non-zero, and an
    unmasked pool would let the board-size fraction leak into the magnitudes.

    The head always emits all `n_players` channels; alive-masking (real,
    currently-alive players only) is applied at loss time, not here.
    """

    def __init__(
        self,
        in_ch: int,
        n_bins: int,
        n_players: int = 8,
        pool: str = "mean",
        hidden: int = 0,
    ):
        super().__init__()
        self.n_players = n_players
        self.n_bins = n_bins
        self.pool = pool
        if hidden > 0:
            self.pre: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            conv_in = hidden
        else:
            self.pre = nn.Identity()
            conv_in = in_ch
        self.conv = nn.Conv2d(conv_in, n_players * n_bins, kernel_size=3, padding=1)
        if pool == "lse":
            # softplus(raw_beta) = temperature β. Init β ≈ 0.5 so the pool starts
            # close to mean and is free to flatten toward mean (β→0) or sharpen
            # toward max (β→∞) as the elim gradient dictates.
            self.raw_beta = nn.Parameter(
                torch.tensor(math.log(math.expm1(0.5)), dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W], valid_mask: [B, 1, H, W] bool → [B, n_players, n_bins]."""
        z = self.conv(self.pre(x))                          # [B, P·n_bins, H, W]
        if self.pool == "mean":
            m = valid_mask.to(x.dtype)
            summed = (z * m).sum(dim=(2, 3))                # [B, P·n_bins]
            count = m.sum(dim=(2, 3)).clamp(min=1.0)        # [B, 1]
            pooled = summed / count
        else:  # "lse" — masked, length-normalized log-sum-exp (smooth max)
            beta = F.softplus(self.raw_beta).to(torch.float32)
            mb = valid_mask.flatten(2).bool()               # [B, 1, H·W]
            # Mask in exponent-space (post-β) with a finite sentinel, NOT -inf on
            # z: a -inf cell feeds 0·(-inf)=NaN into β's gradient through
            # logsumexp. A β-independent -1e9 underflows to weight 0 with finite
            # grad and no β leak as β→0.
            s = (beta * z.flatten(2).float()).masked_fill(~mb, -1e9)
            lse = torch.logsumexp(s, dim=2)                 # [B, P·n_bins]
            log_n = mb.sum(dim=2).clamp(min=1).float().log()  # [B, 1]
            pooled = ((lse - log_n) / beta).to(x.dtype)     # [B, P·n_bins]
        return pooled.view(pooled.shape[0], self.n_players, self.n_bins)
