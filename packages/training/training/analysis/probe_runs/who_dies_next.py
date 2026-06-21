#!/usr/bin/env -S uv run python
"""One-off: who-dies-next victim-target frozen-trunk probe (docs 6.14).

`lowest_army` showed the trunk linearly preserves lowest-army; this is the
direct follow-up — can a fresh head on the frozen trunk beat the lowest-army
RULE at predicting the ACTUAL next-death victim? The dataset already labels the
victim (`next_elim_target`), so this reuses the shared cross-player heads +
masked CE and only swaps the target.

The headline is `top1_imminent` (dt<10): the aggregate is buoyed by the easy
2-3-alive endgame, so the contested imminent regime is where the rule actually
has to be beaten (6.13-15). `--dt-max` restricts *training and eval* to the
near-term slice (targets beyond dt are masked to the -1 ignore index), isolating
that regime so the head is not spending capacity on the easy endgame:

  --dt-max unset  -> aggregate: every victim frame contributes.
  --dt-max 10     -> train+eval on dt<10 only (the contested slice).
  --dt-max 25     -> a wider near-term slice (~2.5x the dt<10 frames) — a
                     data-volume control for the dt<10 run.

`top1_imminent` is always the dt<10 sub-slice regardless of `--dt-max`, so runs
are comparable on the same yardstick. "Early-stop" is free — the summary reports
best-val, which matters because the joint head's best val was ep2 before it
overfit.

  beats the rule (esp. imminent) -> the trunk carries learnable residual beyond
    the leaderboard; the joint head's overfit was the problem (regularize / early-stop).
  caps at ~the rule even here -> who-dies-next ~= the leaderboard, the head was
    near-optimal, and its unique trunk contribution is ~nil (weakens the
    24k-transfer bet).
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn

from training.analysis.probes.cli import build_probe_arg_parser, run_probe_cli
from training.analysis.probes.core import FrameNeeds
from training.analysis.probes.cross_player import (
    cross_player_head_zoo,
    masked_cross_player_ce,
    masked_top1,
)
from utils.docstring import doc_summary


class WhoDiesNextTask:
    """Decode the actual next-death victim (`next_elim_target`) from a frozen
    representation, as a cross-player softmax over the alive field. `dt_max`, when
    set, masks targets outside the dt<dt_max slice so the head trains and is
    evaluated only on the near-term regime; left None, every victim frame counts."""

    baselines = {
        "rule_agg": 0.471,
        "rule_imminent": 0.478,
        "oracle": 0.562,
        "uniform": 0.286,
    }
    frame_needs = FrameNeeds(elim_variant="next_death")

    def __init__(self, dt_max: int | None = None):
        self.dt_max = dt_max
        self.name = "who_dies_next" if dt_max is None else f"who_dies_next_dt{dt_max}"

    def extract_target(self, frame: dict[str, Tensor]) -> dict[str, Tensor]:
        dt = frame["next_elim_dt"]                       # horizon; -1 = no future death
        if self.dt_max is None or 0 <= int(dt) < self.dt_max:
            target = frame["next_elim_target"]           # the real victim, -1 = none
        else:
            target = torch.tensor(-1, dtype=torch.int64)  # outside the slice → ignored
        return {
            "target": target,
            "valid_mask": frame["valid_mask"],
            # next_death targets board-removal over the present domain, so the
            # cross-player softmax masks to present_mask (not alive_mask) — else a
            # surrendered-but-present victim is -inf and the label is unreachable.
            "present_mask": frame["present_mask"],
            "dt": dt,
        }

    def build_heads(self, in_ch: int, H: int, W: int) -> dict[str, nn.Module]:
        return cross_player_head_zoo(in_ch)

    def loss(self, pred: Tensor, target: Tensor, aux: dict) -> Tensor:
        return masked_cross_player_ce(pred, target, aux["present_mask"])

    def metrics(self, pred: Tensor, target: Tensor, aux: dict) -> dict[str, float]:
        dt = aux["dt"]
        imm = (dt >= 0) & (dt < 10)
        return {
            "top1": masked_top1(pred, target, aux["present_mask"]),
            "top1_imminent": masked_top1(pred[imm], target[imm], aux["present_mask"][imm]),
        }


def main() -> None:
    parser = build_probe_arg_parser(doc_summary(__doc__))
    parser.add_argument(
        "--dt-max", type=int, default=None,
        help="Restrict train+eval to the dt<DT_MAX imminent slice (targets beyond "
        "it are masked out). Default: all victim frames (aggregate).",
    )
    args = parser.parse_args()
    run_probe_cli(lambda _cfg: WhoDiesNextTask(dt_max=args.dt_max), args)


if __name__ == "__main__":
    main()
