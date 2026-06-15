#!/usr/bin/env -S uv run python
"""One-off: lowest-army identity frozen-representation probe (docs 6.13-15 / 6.14).

Decode "which alive player has the lowest total army" from a frozen
representation. Same shape as the deployed next-death head (a cross-player
softmax over the alive field), but targeting the lowest-army rule directly. The
who-dies-next post-mortem turns on whether the trunk preserved the army
leaderboard: the trained next-death head lands *below* the lowest-army rule even
though per-player army is a broadcast obs channel, so a linear readout of the
inputs expresses the rule by construction. If a linear readout of the *frozen
trunk* recovers lowest-army, the trunk kept the signal; if not, it mangled
something the obs hands it. See 6.14 discussion.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn

from training.analysis.probes.cli import build_probe_arg_parser, run_probe_cli
from training.analysis.probes.core import FrameNeeds
from training.analysis.probes.cross_player import (
    army_totals_ch_indices,
    cross_player_head_zoo,
    masked_cross_player_ce,
    masked_top1,
)
from utils.docstring import doc_summary


class LowestArmyTask:
    """Cross-player softmax targeting argmin-army over the alive field. If a
    linear readout of the trunk reaches ~47% (the model-free rule's accuracy),
    the trunk preserves army; if not, it mangled a signal the obs hands it."""

    name = "lowest_army"
    # The model-free lowest-army rule and uniform-over-alive, on the val split
    # (docs 6.13-15 / 6.14). Drawn as reference lines.
    baselines = {"lowest_army_rule": 0.471, "uniform": 0.284}
    frame_needs = FrameNeeds(alive_mask=True)  # the softmax domain; target is self-derived

    def __init__(self, dense_history_n: int):
        self.army_ch = army_totals_ch_indices(dense_history_n)

    def extract_target(self, frame: dict[str, Tensor]) -> dict[str, Tensor]:
        obs = frame["obs"]                          # [C, H, W]
        valid = frame["valid_mask"]                 # [1, H, W] bool
        alive = frame["alive_mask"]                 # [8] bool
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
        return cross_player_head_zoo(in_ch)

    def loss(self, pred: Tensor, target: Tensor, aux: dict) -> Tensor:
        return masked_cross_player_ce(pred, target, aux["alive_mask"])

    def metrics(self, pred: Tensor, target: Tensor, aux: dict) -> dict[str, float]:
        return {"top1": masked_top1(pred, target, aux["alive_mask"])}


def main() -> None:
    args = build_probe_arg_parser(doc_summary(__doc__)).parse_args()
    run_probe_cli(lambda cfg: LowestArmyTask(dense_history_n=cfg.dense_history_n), args)


if __name__ == "__main__":
    main()
