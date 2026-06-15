"""Reusable kit for "pick a player" probes — the cross-player softmax family.

PROVISIONAL (2026-06-15) — still a bit of a grab-bag and the grouping isn't
settled. Not everything here is "cross-player": `masked_top1` is generic, and the
head zoo is really a per-player *pooled readout* (the army-regression head is the
same shape), not softmax-specific. Only `masked_cross_player_ce` is intrinsically
cross-player. Likely wants a rename or a split (a generic per-player-readout kit
vs the softmax loss/metric). Don't treat the current home of these as
load-bearing. (The obs-channel introspection that used to live here moved to
`analysis/obs_utils.py`.)

A cross-player probe decodes *which player* (a categorical over the alive field)
from a frozen representation: who dies next, who has the lowest army, etc. Every
such probe shares the same readout shape, loss, and metric — they differ only in
the per-frame target. That shared machinery lives here so a one-off in
`probe_runs/` is just an `extract_target` plus a thin `main`.

What's reusable across the family:

  - The head zoo (`ConvMeanPoolHead` / `DeployedElimHead` / `ConvMLPPoolHead`) —
    a fixed linear → deployed-shape → fat spread, so capacity is probed the same
    way everywhere and results stay comparable across probes.
  - `masked_cross_player_ce` / `masked_top1` — the alive-masked CE and top-1 that
    mirror the real next-death head's training loss (`bc/loss.py`). Single-sourced
    here on purpose: copied into each one-off they would drift out of sync with
    the loss they are meant to match.

This kit is family-scoped, not universal: a regression probe (e.g. per-player
army R²) wants a different head/loss/metric and would build its own, on top of
the generic `probes/core.py` framework rather than this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.analysis.obs_utils import N_PLAYERS
from training.bc.model.heads.elim_next_death import ElimNextDeathHead
from training.bc.model.heads.pool import masked_mean_pool


# ---------------------------------------------------------------------------
# Head variants — all take (feats[B,C,H,W], aux) and return [B, N_PLAYERS] logits
# ---------------------------------------------------------------------------
#
# The spread is deliberate: a linear readout, the deployed head's exact shape,
# and a fat (2-conv) readout. Comparing them localizes a failure — linear works
# → the signal is pool-readable; only fat works → it's present but entangled;
# neither → the representation doesn't carry it decodably at this scale.


class ConvMeanPoolHead(nn.Module):
    """Linear readout: one conv → masked mean pool. Mean pool is exact for a
    broadcast (per-player constant) level signal, so this is the cleanest test
    of "is the target linearly present"."""

    def __init__(self, in_ch: int, n_players: int = N_PLAYERS):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_players, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, aux: dict) -> torch.Tensor:
        return masked_mean_pool(self.conv(x), aux["valid_mask"])


class DeployedElimHead(nn.Module):
    """The real `next_death` head's exact shape (conv → masked lse pool with a
    learnable temperature). Adapts its `(x, valid_mask)` signature to the
    probe's `(x, aux)` calling convention."""

    def __init__(self, in_ch: int, n_players: int = N_PLAYERS):
        super().__init__()
        self.head = ElimNextDeathHead(in_ch=in_ch, n_players=n_players)

    def forward(self, x: torch.Tensor, aux: dict) -> torch.Tensor:
        return self.head(x, aux["valid_mask"])


class ConvMLPPoolHead(nn.Module):
    """Fat readout: two convs with a GELU between, then masked mean pool. The
    capacity probe — if only this recovers the signal, it's present but needs
    nonlinear disentangling."""

    def __init__(self, in_ch: int, n_players: int = N_PLAYERS, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, n_players, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, aux: dict) -> torch.Tensor:
        return masked_mean_pool(self.net(x), aux["valid_mask"])


def cross_player_head_zoo(in_ch: int) -> dict[str, nn.Module]:
    """The standard linear / deployed / fat readouts a cross-player probe tries.
    A `build_heads` is usually just `return cross_player_head_zoo(in_ch)`."""
    return {
        "linear_pool": ConvMeanPoolHead(in_ch),
        "deployed": DeployedElimHead(in_ch),
        "fat": ConvMLPPoolHead(in_ch),
    }


# ---------------------------------------------------------------------------
# Shared masked cross-player CE / top-1 (mirrors training/bc/loss.py)
# ---------------------------------------------------------------------------


def masked_cross_player_ce(
    logits: torch.Tensor, target: torch.Tensor, alive_mask: torch.Tensor
) -> torch.Tensor:
    """Cross-player CE over the alive field only, `-1` targets ignored — the
    exact form the real next-death head trains under (loss.py:314-327)."""
    masked = logits.masked_fill(~alive_mask, float("-inf"))
    n = (target != -1).sum().clamp(min=1)
    return F.cross_entropy(masked, target, ignore_index=-1, reduction="sum") / n


def masked_top1(
    logits: torch.Tensor, target: torch.Tensor, alive_mask: torch.Tensor
) -> float:
    masked = logits.masked_fill(~alive_mask, float("-inf"))
    valid = target != -1
    if valid.sum() == 0:
        return 0.0
    return (masked.argmax(dim=1)[valid] == target[valid]).float().mean().item()
