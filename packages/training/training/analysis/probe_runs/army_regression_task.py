#!/usr/bin/env -S uv run python
"""One-off: per-player army-regression frozen-representation probe (docs 6.14-1 Open 1).

Regress each alive player's total army off a frozen representation and report
per-player R². This is the continuous companion to the lowest-army identity probe
(`lowest_army.py`), which saturated (~0.6) and so couldn't separate "the trunk
preserves army cleanly" from "the trunk preserves it degraded". R² has no such
ceiling: raw-obs lands ≈1.0 by construction (army is a broadcast obs channel), so
that run is the calibration anchor, and a trunk's per-player R² as a fraction of
it answers whether the multi-task training kept the army leaderboard or washed it
out.

Masked to alive players: a dead player's army is ≈0 and trivially predictable,
which would inflate R². Linear head only — masked mean pool is exact for a
broadcast (per-player constant) signal, so a one-conv readout is the right
instrument; capacity isn't the question here.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn

from training.analysis.obs_utils import N_PLAYERS, army_totals_ch_indices, per_player_army
from training.analysis.probes.cli import build_probe_arg_parser, run_probe_cli
from training.analysis.probes.core import FrameNeeds
from training.bc.model.heads.pool import masked_mean_pool
from utils.docstring import doc_summary


class ArmyRegressionHead(nn.Module):
    """Linear readout: one conv → masked mean pool → the per-player army scalars.
    Mean pool is exact for a broadcast level signal, so this is the clean test of
    "is the army leaderboard linearly present in the representation". Same shape
    as the cross-player pooled readout (`ConvMeanPoolHead`); kept local while that
    pooled-readout kit is still provisional (see `probes/cross_player.py`)."""

    def __init__(self, in_ch: int, n_players: int = N_PLAYERS):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_players, kernel_size=3, padding=1)

    def forward(self, x: Tensor, aux: dict) -> Tensor:
        return masked_mean_pool(self.conv(x), aux["valid_mask"])


class ArmyRegressionTask:
    """Regress the per-player total-army scalars over the alive field. Per-player
    R² (and their mean) vs the raw-obs anchor say whether the trunk kept army."""

    name = "army_regression"
    baselines: dict[str, float] = {}  # no model-free constant; the raw-obs run is the anchor
    frame_needs = FrameNeeds(alive_mask=True)  # the regression domain; target is self-derived

    def __init__(self, dense_history_n: int):
        self.army_ch = army_totals_ch_indices(dense_history_n)

    def extract_target(self, frame: dict[str, Tensor]) -> dict[str, Tensor]:
        obs = frame["obs"]                          # [C, H, W]
        valid = frame["valid_mask"]                 # [1, H, W] bool
        alive = frame["alive_mask"]                 # [8] bool
        army = per_player_army(obs, valid, self.army_ch).float()  # [8]
        return {"target": army, "valid_mask": valid, "alive_mask": alive}

    def build_heads(self, in_ch: int, H: int, W: int) -> dict[str, nn.Module]:
        return {"linear": ArmyRegressionHead(in_ch)}

    def loss(self, pred: Tensor, target: Tensor, aux: dict) -> Tensor:
        # Mean squared error over alive entries only (multiply-then-normalize so a
        # frame with no alive players can't divide by zero).
        alive = aux["alive_mask"].float()           # [B, 8]
        se = (pred - target) ** 2 * alive
        return se.sum() / alive.sum().clamp(min=1)

    def metrics(self, pred: Tensor, target: Tensor, aux: dict) -> dict[str, float]:
        # Whole-split per-player R²: for each player, accumulate residual / total
        # sums of squares over that player's alive frames, then take the ratio. The
        # eval loop hands us the full val split, so this is the exact reduction (not
        # a per-frame R² averaged). Player 0 = self, 1..7 = opponents.
        alive = aux["alive_mask"].bool()            # [N, 8]
        per_player: dict[str, float] = {}
        valid_r2: list[float] = []
        for p in range(N_PLAYERS):
            m = alive[:, p]
            t, pr = target[m, p], pred[m, p]
            ss_tot = ((t - t.mean()) ** 2).sum() if t.numel() >= 2 else target.new_zeros(())
            if ss_tot > 0:
                r2 = (1 - ((pr - t) ** 2).sum() / ss_tot).item()
                valid_r2.append(r2)
            else:
                r2 = 0.0  # degenerate (a player never alive / constant army) — unreachable on real splits
            per_player[f"r2_p{p}"] = r2
        r2_mean = sum(valid_r2) / len(valid_r2) if valid_r2 else 0.0
        return {"r2_mean": r2_mean, **per_player}


def main() -> None:
    args = build_probe_arg_parser(doc_summary(__doc__)).parse_args()
    run_probe_cli(lambda cfg: ArmyRegressionTask(dense_history_n=cfg.dense_history_n), args)


if __name__ == "__main__":
    main()
