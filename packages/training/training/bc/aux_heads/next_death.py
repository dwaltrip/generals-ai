"""The `next_death` (who-is-removed-next) elimination aux head, as a spec.

A single cross-player softmax per frame over "which present player is removed
next", on the board-removal/`present_mask` domain, with an optional removal-time
soft target. This spec is the home of the variant's targets, loss (the
who-dies-next CE + soft target), and dump columns; the head module is
`model/heads/elim_next_death.py`. No in-loop eval meter — its diagnostics are
computed offline from the dump. See
`docs/2026-06/6.21-1-aux-head-spec-registry-refactor.md`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.bc.aux_heads.base import AuxLossResult
from training.bc.targets.elim_targets import ElimCtx, next_death_target


class NextDeathSpec:
    name = "next_death"
    output_key = "next_elim_logits"
    metric_keys: tuple[str, ...] = ("next_elim", "next_elim_soft")
    term_key = "next_elim_soft"
    count_key = "n_next_elim"
    weight_attr = "lambda_elim"

    def build_head(self, arch: Any) -> nn.Module:
        from training.bc.model.heads.elim_next_death import ElimNextDeathHead

        return ElimNextDeathHead(in_ch=arch.outer_width)

    def encode_targets(
        self, ctx: ElimCtx, raw_order: list[int], t: int
    ) -> dict[str, torch.Tensor]:
        nxt, present, dt, removal_dt = next_death_target(ctx, raw_order, t)
        # next_death targets board-removal over the *present* domain (not the
        # alive domain), so it carries its own `present_mask`; the loss/dump
        # softmax over it, not `alive_mask`.
        return {
            "present_mask": torch.from_numpy(present),
            "next_elim_target": torch.tensor(nxt, dtype=torch.int64),
            "next_elim_dt": torch.tensor(dt, dtype=torch.int64),
            "next_elim_removal_dt": torch.from_numpy(removal_dt),
        }

    def loss(
        self,
        model_out: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        cfg: Any,
    ) -> AuxLossResult:
        nd_logits = model_out["next_elim_logits"]       # [B, 8]
        nd_target = targets["next_elim_target"]         # [B] int64, -1 = ignore
        present_mask = targets["present_mask"]          # [B, 8] bool
        # Cross-player softmax over the present field only: removed/phantom
        # channels get -inf logits → zero prob, zero gradient. Channel 0 (self) is
        # always present in-trajectory, so no included row is all-masked.
        # ignore_index=-1 drops the winner-tail frames that carry no next removal;
        # reduction="sum" over the kept frames / their count mirrors the policy-CE
        # safe divide.
        masked_logits = nd_logits.masked_fill(~present_mask, float("-inf"))
        valid = nd_target != -1                         # [B] frames with a real next removal
        n_next_elim = valid.sum()                       # 0-d, stays on device
        denom = n_next_elim.clamp(min=1)
        # Hard CE against the one-hot next victim — always the cross-run-comparable
        # reporting metric (the top-1 the heuristics baseline against).
        next_elim = F.cross_entropy(
            masked_logits, nd_target, ignore_index=-1, reduction="sum"
        ) / denom
        if cfg.next_elim_target_tau > 0:
            # Soft objective: a per-frame distribution over present players,
            # `p_i ∝ exp(-(removal_dt_i - removal_dt_min)/τ)`. softmax subtracts the
            # row-max (the `-removal_dt_min` term) for free, and the winner's huge
            # `removal_dt` underflows to ~0 mass — no explicit min/winner handling.
            removal_dt = targets["next_elim_removal_dt"].to(torch.float32)   # [B, 8]
            neg = torch.where(
                present_mask, -removal_dt / cfg.next_elim_target_tau, float("-inf")
            )
            soft_target = F.softmax(neg, dim=1)                             # [B, 8]
            # CE against prob targets has no ignore_index, so mask by hand. Zero
            # log-prob on absent channels first: soft_target is 0 there, but the
            # logits are -inf → log_softmax -inf, and 0·(-inf)=nan would poison it.
            logp = F.log_softmax(masked_logits, dim=1)
            logp = torch.where(present_mask, logp, torch.zeros_like(logp))
            ce = -(soft_target * logp).sum(dim=1)                          # [B]
            next_elim_soft = torch.where(
                valid, ce, torch.zeros_like(ce)
            ).sum() / denom
        else:
            next_elim_soft = next_elim

        return AuxLossResult(
            metrics={"next_elim": next_elim, "next_elim_soft": next_elim_soft},
            count=n_next_elim,
        )

    def dump_records(
        self,
        out: dict[str, torch.Tensor],
        moved_batch: dict[str, torch.Tensor],
        host_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Cross-player softmax over the *present* field + per-frame hard CE.
        # Masked to present players (removed/phantom → -inf → prob 0); winner-tail
        # frames (target -1) get NaN CE (nan-aware reductions downstream).
        present_mask = moved_batch["present_mask"]
        nd_logits = out["next_elim_logits"].float().masked_fill(
            ~present_mask, float("-inf")
        )
        nd_logp = F.log_softmax(nd_logits, dim=1)       # [B, 8]
        nd_target = moved_batch["next_elim_target"]     # [B]; -1 = none
        nd_ce = -nd_logp.gather(
            1, nd_target.clamp(min=0).unsqueeze(1)
        ).squeeze(1)
        nd_ce = torch.where(
            nd_target < 0, torch.full_like(nd_ce, float("nan")), nd_ce
        )
        # `alive_mask` rides along too (the dataset emits it unconditionally):
        # offline tooling keyed on the alive domain stays joinable, and the
        # alive-vs-present gap (the surrender window) is itself analyzable.
        return {
            "next_elim_probs": nd_logp.exp().to(torch.float16),
            "next_elim_ce": nd_ce,
            "next_elim_target": host_batch["next_elim_target"],
            "present_mask": host_batch["present_mask"],
            "alive_mask": host_batch["alive_mask"],
            "next_elim_dt": host_batch["next_elim_dt"],
        }

    def build_eval_meter(self, cfg: Any, edges: tuple[int, ...] | None) -> Any | None:
        # No in-loop meter — next_death diagnostics are computed offline from the
        # dump's raw columns.
        return None

    def eval_update(
        self,
        meter: Any,
        out: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> None:
        return None

    def eval_summary(
        self, meter: Any, accum_summary: dict[str, float | int]
    ) -> dict[str, float | None]:
        # Only the loss metric passes through to the eval summary; no meter reads.
        if "next_elim" in accum_summary:
            return {"next_elim": accum_summary.get("next_elim")}
        return {}

    def validate(self, arch: Any, cfg: Any) -> None:
        # The next_death soft target only exists for this variant — a τ set against
        # another variant (or no head) would silently never apply. Checked across
        # all registry specs (not just the active one), so it catches a τ aimed at
        # the wrong variant.
        if cfg.next_elim_target_tau > 0 and arch.elim_head_variant != self.name:
            raise ValueError(
                "next_elim_target_tau > 0 requires "
                'arch.elim_head_variant == "next_death" '
                f"(got {arch.elim_head_variant!r})"
            )
