"""Who-dies-next elimination head — a single cross-player softmax per frame.

The `next_death` variant of the elimination aux head (see `ElimTimeBinHead` for
the `time_bin` variant). Per frame, predicts **which player is eliminated next**:
one logit per canonical player channel (0 = perspective/self, 1..7 = opponents in
`opp_slots` order), normalized by a softmax *across players* into a single
categorical over the currently-alive field. The target redesign of 6.13-12 /
6.13-13: it drops the noisy absolute time-to-death and points only at the order
of the next elimination — the part the current state actually determines —
forcing the cross-player comparison the identity-agnostic time-bin head never
pressured. See `docs/2026-06/6.13-12-who-dies-next-target-reframe.md` and
`docs/2026-06/6.13-13-who-dies-next-poc-plan.md`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.bc.model.heads.pool import masked_lse_pool


class ElimNextDeathHead(nn.Module):
    """Per-player "dies-next" logits pooled from the trunk embedding.

    Structure (a single-output-per-player slimming of `ElimTimeBinHead`):

        x ─→ Conv2d(C → n_players, k=3, pad=1)  # per-cell per-player evidence
          ─→ masked lse pool over the unpadded board
          ─→ [B, n_players]

    The pool is the masked, length-normalized log-sum-exp (a smooth max with a
    learnable temperature β) — who-dies-next is partly a localized-danger
    detector ("player i under attack at these few tiles"), and lse preserves that
    spatial peak where the mean pool dilutes it (6.13-11). Pool choice is fixed
    to lse for now; a `pool` knob can be threaded later if mean is wanted.

    No ReLU on the readout conv: we pool straight to logits, and a ReLU would
    discard signed evidence. The mask is required for the same reason as the
    other pooled heads' — trunk activations at padded cells are non-zero, and an
    unmasked pool would let the board-size fraction leak into the magnitudes.

    The head always emits all `n_players` logits; the cross-player softmax is
    taken over the *alive* players only, and that alive-masking is applied at
    loss/eval time (the head has no aliveness input), mirroring `ElimTimeBinHead`.
    """

    def __init__(self, in_ch: int, n_players: int = 8):
        super().__init__()
        self.n_players = n_players
        self.conv = nn.Conv2d(in_ch, n_players, kernel_size=3, padding=1)
        # softplus(raw_beta) = temperature β. Init β ≈ 0.5 so the pool starts
        # close to mean and is free to flatten toward mean (β→0) or sharpen
        # toward max (β→∞) as the gradient dictates.
        self.raw_beta = nn.Parameter(
            torch.tensor(math.log(math.expm1(0.5)), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W], valid_mask: [B, 1, H, W] bool → [B, n_players]."""
        z = self.conv(x)                                # [B, n_players, H, W]
        beta = F.softplus(self.raw_beta).to(torch.float32)
        return masked_lse_pool(z, valid_mask, beta).to(x.dtype)
