"""Concrete probe tasks + the head variants they share.

Each task is a `core.ProbeTask`: it defines the per-frame target, a small zoo of
swap-in heads spanning capacities (linear → deployed-shape → fat), the loss, the
metrics, and the model-free baselines to draw as reference lines.

`LowestArmyTask` is the firm read for the who-dies-next post-mortem (docs
6.13-15 / 6.14): the trained next-death head lands *below* the 47% lowest-army
rule even though per-player army is a broadcast obs channel, so a linear readout
of the inputs expresses the rule by construction. Probing whether a *frozen
trunk's* features still let a linear readout recover lowest-army separates "the
trunk discarded the signal" from "the head never learned to read a signal that
is present" — see 6.14 discussion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.bc.constants import obs_channel_names
from training.bc.model.heads.elim_next_death import ElimNextDeathHead
from training.bc.model.heads.pool import masked_mean_pool

N_PLAYERS = 8


def army_channel_indices(dense_history_n: int) -> list[int]:
    """Obs channel indices of the per-player total-army planes, canonical player
    order (0 = self, 1..7 = opponents in `opp_slots` order) — aligned with the
    elim head's player axis and `next_elim_alive_mask`. These planes are
    broadcast scalars (`army / 1000`), fog-independent, in the fixed base
    channels, so the indices are independent of `dense_history_n`.
    """
    names = obs_channel_names(dense_history_n)
    wanted = ["self_army_count"] + [f"opp_{i}_army_count" for i in range(1, N_PLAYERS)]
    return [names.index(w) for w in wanted]


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
    of "is lowest-army linearly present"."""

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


# ---------------------------------------------------------------------------
# Shared masked cross-player CE / top-1 (mirrors training/bc/loss.py)
# ---------------------------------------------------------------------------


def masked_cross_player_ce(
    logits: torch.Tensor, target: torch.Tensor, alive_mask: torch.Tensor
) -> torch.Tensor:
    """Cross-player CE over the alive field only, `-1` targets ignored — the
    exact form the real next-death head trains under (loss.py:323-327)."""
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


# ---------------------------------------------------------------------------
# LowestArmyTask
# ---------------------------------------------------------------------------


class LowestArmyTask:
    """Decode "which alive player has the lowest total army" from the frozen
    representation. Same shape as the deployed next-death head (a cross-player
    softmax over the alive field), targeting the lowest-army rule directly. If a
    linear readout of the trunk reaches ~47% (the model-free rule's accuracy),
    the trunk preserves army; if not, it mangled a signal the obs hands it.
    """

    name = "lowest_army"
    primary_metric = "top1"
    # The model-free lowest-army rule and uniform-over-alive, on the val split
    # (docs 6.13-15 / 6.14). Drawn as reference lines.
    baselines = {"lowest_army_rule": 0.471, "uniform": 0.284}
    elim_variant = "next_death"  # for next_elim_alive_mask (the softmax domain)

    def __init__(self, dense_history_n: int):
        self.army_ch = army_channel_indices(dense_history_n)

    def extract_target(self, frame: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obs = frame["obs"]                          # [C, H, W]
        valid = frame["valid_mask"]                 # [1, H, W] bool
        alive = frame["next_elim_alive_mask"]       # [8] bool
        # Recover each player's army scalar: the planes are constant over the
        # real board, so the valid-masked mean is exactly that value.
        planes = obs[self.army_ch].float()          # [8, H, W]
        m = valid.float()
        army = (planes * m).sum(dim=(-1, -2)) / m.sum().clamp(min=1)  # [8]
        if bool(alive.any()):
            big = torch.where(alive, army, torch.full_like(army, float("inf")))
            target = int(big.argmin())
        else:
            target = -1  # no alive field (terminal frame) — ignored in loss
        return {
            "target": torch.tensor(target, dtype=torch.int64),
            "valid_mask": valid,
            "alive_mask": alive,
        }

    def build_heads(self, in_ch: int, H: int, W: int) -> dict[str, nn.Module]:
        return {
            "linear_pool": ConvMeanPoolHead(in_ch),
            "deployed": DeployedElimHead(in_ch),
            "fat": ConvMLPPoolHead(in_ch),
        }

    def loss(self, pred: torch.Tensor, target: torch.Tensor, aux: dict) -> torch.Tensor:
        return masked_cross_player_ce(pred, target, aux["alive_mask"])

    def metrics(self, pred: torch.Tensor, target: torch.Tensor, aux: dict) -> dict[str, float]:
        return {"top1": masked_top1(pred, target, aux["alive_mask"])}


TASKS: dict[str, type] = {
    "lowest_army": LowestArmyTask,
}
