#!/usr/bin/env -S uv run python
"""One-off: regress-then-argmin frozen-representation probe (docs 6.16-x).

The diagnostic this isolates: is the who-dies-next head's shortfall the *army
info* or *learning argmin*? Two probes bracket it, sharing one readout shape
(conv → masked mean pool → 8):

  - LowestArmyTask (6.14-1) trains that head on the hard argmin target via
    cross-player CE and reads argmax of the logits → ~0.576 argmin-identity off
    the trunk (and, tellingly, only ~0.53–0.61 even off raw-obs, which holds the
    exact army).
  - This probe trains the *same* head on the continuous army target (the proven
    0.82-R² MSE objective) and takes argmin of the predicted armies — exact
    arithmetic, not a learned softmax.

If regress-then-argmin ≫ 0.576, the army info was there and the bottleneck is
learning argmin via cross-player CE (actionable: the elim head's loss/target).
If it lands ≈0.6 too, then 0.82-R² army genuinely caps the argmin. The raw-obs
cell is the clean tell: regress-then-argmin should hit ~1.0 there, whereas
learned-argmin saturated ~0.6 — that gap would prove the wall is a learning
artifact, not fundamental.

Subclasses ArmyRegressionTask to keep that documented experiment untouched: the
head, target, and MSE loss are inherited verbatim (so r2_mean reproduces 0.82 as
a sanity check), and only the argmin-identity metric and the task name are added.
"""

from __future__ import annotations

from torch import Tensor

from training.analysis.probe_runs.army_regression_task import ArmyRegressionTask
from training.analysis.probes.cli import build_probe_arg_parser, run_probe_cli
from utils.docstring import doc_summary


class RegressThenArgminTask(ArmyRegressionTask):
    """Army regression, additionally scored by the argmin of its predicted
    leaderboard. `argmin_identity` is directly comparable to LowestArmyTask's
    learned-argmin top-1 (~0.576); `r2_mean` reproduces army-regression's 0.82."""

    name = "regress_then_argmin"

    def metrics(self, pred: Tensor, target: Tensor, aux: dict) -> dict[str, float]:
        out = super().metrics(pred, target, aux)
        # Argmin of the *predicted* armies over the alive field vs the true
        # lowest-army player. Fill dead slots with +inf so they're never the
        # argmin; skip terminal frames with no alive field.
        alive = aux["alive_mask"].bool()            # [N, 8]
        has_alive = alive.any(dim=1)
        pred_low = pred.masked_fill(~alive, float("inf")).argmin(dim=1)
        true_low = target.masked_fill(~alive, float("inf")).argmin(dim=1)
        out["argmin_identity"] = (
            (pred_low[has_alive] == true_low[has_alive]).float().mean().item()
            if bool(has_alive.any())
            else 0.0
        )
        return out


def main() -> None:
    args = build_probe_arg_parser(doc_summary(__doc__)).parse_args()
    run_probe_cli(lambda cfg: RegressThenArgminTask(dense_history_n=cfg.dense_history_n), args)


if __name__ == "__main__":
    main()
