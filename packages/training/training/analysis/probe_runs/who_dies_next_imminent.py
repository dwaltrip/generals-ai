#!/usr/bin/env -S uv run python
from __future__ import annotations

import torch
import torch.nn as nn

from training.analysis.probes.cli import build_probe_arg_parser, run_probe_cli
from training.analysis.probes.tasks import (
    ConvMeanPoolHead,
    ConvMLPPoolHead,
    DeployedElimHead,
    masked_cross_player_ce,
    masked_top1,
)
from utils.docstring import doc_summary


class WhoDiesNextImminentTask:
    """Decode the actual next-death victim from a frozen representation. Same
    head zoo / loss / metric as `LowestArmyTask`; the target is the dataset's
    `next_elim_target` instead of argmin-army, so the lowest-army rule is now the
    right line to beat."""

    name = "who_dies_next_imminent"
    primary_metric = "top1_imminent"
    # This task's target IS the victim, so the rule's victim-accuracy is the
    # reference (docs 6.13-15, on the 11k val split). imminent = the dt<10 rule.
    baselines = {"rule_agg": 0.471, "rule_imminent": 0.478, "oracle": 0.562, "uniform": 0.286}
    elim_variant = "next_death"

    def extract_target(self, frame: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        dt = frame["next_elim_dt"]
        # Train + eval on the imminent slice only: everything else is ignored via
        # the -1 sentinel the loss and top-1 already honor. This is the clean test
        # of "can a head focused on dt<10 beat the rule's 0.478 there?"

        # imminent = 0 <= int(dt) < 10
        imminent = 0 <= int(dt) < 25

        target = frame["next_elim_target"] if imminent else torch.tensor(-1, dtype=torch.int64)
        return { 
            "target": target,
            "valid_mask": frame["valid_mask"],
            "alive_mask": frame["next_elim_alive_mask"],
            "dt": dt,
        } 
        # --- version from `who_dies_next.py`:
        # return {
        #     "target": frame["next_elim_target"],        # the real victim, -1 = none
        #     "valid_mask": frame["valid_mask"],
        #     "alive_mask": frame["next_elim_alive_mask"],
        #     "dt": frame["next_elim_dt"],                 # horizon, for the imminent slice
        # }

    def build_heads(self, in_ch: int, H: int, W: int) -> dict[str, nn.Module]:
        return {
            "linear_pool": ConvMeanPoolHead(in_ch),
            "deployed": DeployedElimHead(in_ch),
            "fat": ConvMLPPoolHead(in_ch),
        }

    def loss(self, pred: torch.Tensor, target: torch.Tensor, aux: dict) -> torch.Tensor:
        return masked_cross_player_ce(pred, target, aux["alive_mask"])

    def metrics(self, pred: torch.Tensor, target: torch.Tensor, aux: dict) -> dict[str, float]:
        dt = aux["dt"]
        imm = (dt >= 0) & (dt < 10)
        return {
            "top1": masked_top1(pred, target, aux["alive_mask"]),
            "top1_imminent": masked_top1(pred[imm], target[imm], aux["alive_mask"][imm]),
        }


def main() -> None:
    args = build_probe_arg_parser(doc_summary(__doc__)).parse_args()
    run_probe_cli(lambda _cfg: WhoDiesNextImminentTask(), args)


if __name__ == "__main__":
    main()
