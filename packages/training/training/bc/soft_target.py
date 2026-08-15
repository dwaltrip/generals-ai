"""Shared soft-target kernels for the CE-based heads.
Used by soft-target loss calcs and related metrics.
"""

from __future__ import annotations

from functools import cache

import torch
import torch.nn.functional as F


# reused every batch, hence the cache
@cache
def soft_target_kernel(
    tau: float,
    n_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Soft-target matrix for τ > 0. Row k* is `softmax(-|k - k*|/τ)`.
    Row-indexing with the hard targets yields the per-sample soft distributions.
    The row softmax renormalizes edge rows (rank 1 / rank 8 have one-sided
    neighborhoods) automatically.
    Kernel dimensions are `[n_classes, n_classes]`
    """
    # float32, as autocast routes cross_entropy to fp32
    ranks = torch.arange(n_classes, dtype=torch.float32, device=device)
    dist = (ranks.unsqueeze(0) - ranks.unsqueeze(1)).abs()
    return F.softmax(-dist / tau, dim=1)


# reused every batch, hence the cache
@cache
def elim_weight_tensor(
    # must be a tuple for @cache to work
    bin_weights: tuple[float, ...],
    device: torch.device,
) -> torch.Tensor:
    """Per time-bin CE weight vector for the elim head."""
    # float32, as autocast routes cross_entropy to fp32
    return torch.tensor(bin_weights, dtype=torch.float32, device=device)
