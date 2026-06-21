"""Shared soft-target kernels for the CE-based heads.

Cached, device-keyed tensors used by the value/elim soft-target losses and the
`ElimMeter` soft floor. Lives in its own module (not `loss.py`) so the aux-head
specs can reuse it without importing `loss.py` — `loss.py` imports the aux-head
registry, so the dependency must point one way.
"""

from __future__ import annotations

from functools import cache

import torch
import torch.nn.functional as F


@cache
def _soft_target_kernel(
    tau: float, n_classes: int, device: torch.device
) -> torch.Tensor:
    """The `[n_classes, n_classes]` soft-target matrix for τ > 0: row k* is
    `softmax(-|k - k*|/τ)`. Row-indexing with the hard targets yields the
    per-sample soft distributions. The row softmax renormalizes edge rows
    (rank 1 / rank 8 have one-sided neighborhoods) automatically.

    Cached per (τ, device) — built once, reused every batch. fp32 on
    purpose: autocast routes cross_entropy to fp32, so the targets match.
    """
    ranks = torch.arange(n_classes, dtype=torch.float32, device=device)
    dist = (ranks.unsqueeze(0) - ranks.unsqueeze(1)).abs()
    return F.softmax(-dist / tau, dim=1)


@cache
def _elim_weight_tensor(
    weights: tuple[float, ...], device: torch.device
) -> torch.Tensor:
    """Per-bin CE weight vector for the elim head, as an fp32 device tensor.

    Cached per `(weights, device)` — the tuple is hashable (LossConfig coerces
    it) so it keys the cache directly. fp32 because autocast routes
    cross_entropy to fp32.
    """
    return torch.tensor(weights, dtype=torch.float32, device=device)
