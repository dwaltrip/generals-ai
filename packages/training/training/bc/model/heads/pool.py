"""Masked spatial pools shared by the trunk-reading heads.

Both reductions collapse a per-cell evidence map `[B, K, H, W]` to a per-sample
vector `[B, K]` while ignoring padded cells — trunk activations at padded
positions are non-zero (convs see the zero-padded input), so an unmasked pool
would let the per-game board-size fraction leak into the magnitudes.

  - `masked_mean_pool` — the masked global average (v1 readout): every real
    cell weighted equally. Dilutes spatially-localized evidence (a strong signal
    at ~3 of ~200 tiles is a ~1.5% perturbation of the mean).
  - `masked_lse_pool` — a masked, length-normalized log-sum-exp: a smooth maximum
    with a learnable temperature β (interpolating mean at β→0 and max at β→∞), so
    a single strong cell can surface a localized signal regardless of board size.
    The caller owns β (a learnable `nn.Parameter`); the pool takes it as an arg.
"""

from __future__ import annotations

import torch


def masked_mean_pool(z: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """`z: [B, K, H, W]`, `valid_mask: [B, 1, H, W]` bool → `[B, K]` (z's dtype)."""
    m = valid_mask.to(z.dtype)
    summed = (z * m).sum(dim=(2, 3))                # [B, K]
    count = m.sum(dim=(2, 3)).clamp(min=1.0)        # [B, 1]
    return summed / count


def masked_lse_pool(
    z: torch.Tensor, valid_mask: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """`z: [B, K, H, W]`, `valid_mask: [B, 1, H, W]` bool, `beta`: scalar →
    `[B, K]` (fp32 — the caller casts back to its working dtype).

    `beta` is the softplus-positive temperature; pass `F.softplus(raw_beta)`.
    """
    mb = valid_mask.flatten(2).bool()               # [B, 1, H·W]
    # Mask in exponent-space (post-β) with a finite sentinel, NOT -inf on z: a
    # -inf cell feeds 0·(-inf)=NaN into β's gradient through logsumexp. A
    # β-independent -1e9 underflows to weight 0 with finite grad and no β leak as
    # β→0.
    s = (beta * z.flatten(2).float()).masked_fill(~mb, -1e9)
    lse = torch.logsumexp(s, dim=2)                 # [B, K]
    log_n = mb.sum(dim=2).clamp(min=1).float().log()  # [B, 1]
    return (lse - log_n) / beta                     # [B, K]


__all__ = ["masked_mean_pool", "masked_lse_pool"]
