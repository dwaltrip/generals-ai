"""The `time_bin` elimination aux head, as a spec.

Per-player time-to-elimination over ordinal bins, on the death/`alive_mask`
domain. This spec is the home of the variant's targets, loss, dump columns, and
in-loop diagnostics; the head module is `model/heads/elim_time_bin.py` and the
diagnostic meter is `eval/metrics.py:ElimMeter`. See
`docs/2026-06/6.21-1-aux-head-spec-registry-refactor.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.bc.aux_heads.base import AuxLossResult
from training.bc.soft_target import _elim_weight_tensor, _soft_target_kernel
from training.bc.targets.elim_targets import ElimCtx, time_bin_targets


if TYPE_CHECKING:
    from training.bc.model.output import ModelOut


class TimeBinSpec:
    name = "time_bin"
    output_key = "elim_logits"
    metric_keys: tuple[str, ...] = ("elim", "elim_soft")
    term_key = "elim_soft"
    count_key = "n_elim"
    weight_attr = "lambda_elim"

    def build_head(self, arch: Any) -> nn.Module:
        from training.bc.model.heads.elim_time_bin import ElimTimeBinHead

        return ElimTimeBinHead(
            in_ch=arch.outer_width,
            n_bins=arch.elim_n_bins,
            pool=arch.elim_pool,
            hidden=arch.elim_head_hidden,
        )

    def encode_targets(
        self, ctx: ElimCtx, raw_order: list[int], t: int
    ) -> dict[str, torch.Tensor]:
        bins, _ = time_bin_targets(ctx, raw_order, t)
        return {"elim_bin_target": torch.from_numpy(bins)}

    def loss(
        self,
        model_out: ModelOut,
        targets: dict[str, torch.Tensor],
        cfg: Any,
    ) -> AuxLossResult:
        elim_logits = model_out.aux["elim_logits"]      # [B, 8, n_bins]
        elim_bin_target = targets["elim_bin_target"]    # [B, 8] int64
        alive_mask = targets["alive_mask"]              # [B, 8] bool
        n_bins = elim_logits.shape[2]
        weight = (
            _elim_weight_tensor(cfg.elim_bin_weights, elim_logits.device)
            if cfg.elim_bin_weights is not None
            else None
        )
        # Per-(player, frame) CE with reduction="none" so we apply the alive
        # mask + masked mean ourselves. Hard CE against the bin label is the
        # reporting metric; the soft variant (soft ordinal targets) is the
        # objective at τ>0. Weight, when set, applies to both — so they stay
        # equal at τ=0.
        logits_flat = elim_logits.reshape(-1, n_bins)            # [B·8, n_bins]
        target_flat = elim_bin_target.reshape(-1)                # [B·8]
        ce_hard = F.cross_entropy(
            logits_flat, target_flat, weight=weight, reduction="none"
        ).reshape(elim_logits.shape[:2])                         # [B, 8]
        if cfg.elim_target_tau > 0:
            kernel = _soft_target_kernel(
                cfg.elim_target_tau, n_bins, elim_logits.device
            )
            ce_soft = F.cross_entropy(
                logits_flat, kernel[target_flat], weight=weight, reduction="none"
            ).reshape(elim_logits.shape[:2])
        else:
            ce_soft = ce_hard

        # Masked mean over alive (player, frame) pairs. Channel 0 (self) is
        # always alive in-trajectory, so the denominator is ≥ B — the
        # clamp(min=1) mirrors the policy-CE safe divide as cheap insurance.
        mask = alive_mask.to(ce_hard.dtype)                 # [B, 8]
        n_elim = alive_mask.sum()                           # 0-d, stays on device
        denom = n_elim.clamp(min=1)
        elim = (ce_hard * mask).sum() / denom
        elim_soft = (ce_soft * mask).sum() / denom

        return AuxLossResult(
            metrics={"elim": elim, "elim_soft": elim_soft},
            count=n_elim,
        )

    def dump_records(
        self,
        out: ModelOut,
        moved_batch: dict[str, torch.Tensor],
        host_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Per-(player, frame) bin distribution + hard CE. CE is unweighted (the
        # report wants per-bin unweighted CE; matches the loss's `elim` only when
        # `elim_bin_weights` is None, the current default).
        elim_logp = F.log_softmax(out.aux["elim_logits"].float(), dim=2)
        elim_ce = -elim_logp.gather(
            2, moved_batch["elim_bin_target"].unsqueeze(2)
        ).squeeze(2)
        return {
            "elim_probs": elim_logp.exp().to(torch.float16),
            "elim_ce": elim_ce,
            "elim_bin_target": host_batch["elim_bin_target"],
            "alive_mask": host_batch["alive_mask"],
        }

    def build_eval_meter(self, cfg: Any, edges: tuple[int, ...] | None) -> Any | None:
        if edges is None:
            return None
        from training.bc.eval.metrics import ElimMeter

        return ElimMeter(n_bins=len(edges) + 1, target_tau=cfg.elim_target_tau)

    def eval_update(
        self,
        meter: Any,
        out: ModelOut,
        batch: dict[str, torch.Tensor],
    ) -> None:
        if meter is None:  # no edges → no meter (guarded like eval_summary)
            return
        meter.update(out.aux["elim_logits"], batch["elim_bin_target"], batch["alive_mask"])

    def eval_summary(
        self, meter: Any, accum_summary: dict[str, float | int]
    ) -> dict[str, float | None]:
        if meter is None:  # time_bin always has edges in practice; guard anyway
            return {}
        return {
            "elim_soft": accum_summary.get("elim_soft"),
            "elim": accum_summary.get("elim"),
            "elim_top1": meter.top1(),
            "elim_soft_floor": meter.soft_floor(),
            "elim_pred_entropy": meter.pred_entropy(),
        }

    def validate(self, arch: Any, cfg: Any) -> None:
        # No cross-cutting rule today (the time_bin-requires-edges rule lives at
        # the dataset / model_config boundary where edges are visible).
        return None
