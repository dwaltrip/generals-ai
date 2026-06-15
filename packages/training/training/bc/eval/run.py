"""
Per-epoch validation pass for BC training.

`run_val` walks a val sample list once and reports the diagnostics we
want on the loss curve:

    - **Losses** (policy / value / pass / total): same definitions as
      train, sample-weighted via the shared `LossAccumulator`. Comparable
      epoch-over-epoch and apples-to-apples with train numbers.

    - **Top-1 / top-3 accuracy**: did the model's highest-probability
      legal action(s) match the demonstrated action? Restricted to non-pass
      frames — the policy head's supervision domain. Pass frames have no
      action target (`action_target == -1`) and contribute zero weight.

    - **`pass_acc`**: did the pass head's sign agree with the demonstrated
      pass/act decision? Denominator = all frames (pass head's domain).

    - **`pass_frac`**: observed rate of pass frames in val. A drift signal
      against the pass head's predicted rate (sigmoid(`pass_logit`)) and
      a sanity check on whether the pass head is collapsing.

    - **`policy_entropy`**: mean Shannon entropy (nats) of the masked policy
      softmax over non-pass frames — the same domain as policy CE / top-k.
      Falling across epochs is the policy sharpening; collapse toward 0
      while top-1 stalls is the mode-collapse alarm. The report layer
      renders e^H next to it ("effective number of actions").

    - **`action_dist` / `action_target_dist`**: 8-bucket histograms over
      `(direction, split)` sub-actions — directional bias / mode collapse
      alarm. `action_target_dist` is constant across epochs (it's a
      property of the val set) but echoed in every row for grep-ability
      when comparing the two.

    - **elim head** (present only when the head is active, via `elim_bin_edges`):
      `elim_soft` / `elim` losses plus `elim_top1`, `elim_soft_floor`,
      `elim_pred_entropy` from `ElimMeter`. The health check is
      `elim_soft < elim_soft_floor` (beats the constant-predictor baseline);
      `elim_pred_entropy` is the collapse alarm. Absent keys on non-elim runs.

Architectural notes:
    - Top-k uses the same flat masked layout as `bc_loss` (via
      `flatten_policy_logits`) — one `torch.topk(k=3)` call per batch.
    - Accuracies are `None` (→ JSON `null`) when their denominator is 0.
      Realistically never fires on a meaningful val set, but consistent
      guarding keeps downstream consumers from silently averaging zeros.
    - Sets `model.eval()` for the pass; caller restores `train()` mode.
"""

from __future__ import annotations

from pathlib import Path
import time

import torch

from training.bc.eval.dump import FrameRecordCapture, iter_val_forward
from training.bc.eval.metrics import ActionDistMeter, ElimMeter, PolicyEntropyMeter
from training.bc.loss import (
    DEFAULT_LOSS_CFG,
    LossAccumulator,
    LossConfig,
    bc_loss,
)
from training.bc.model import flatten_policy_logits
from training.bc.obs_config import ObsConfig


def run_val(
    model: torch.nn.Module,
    val_samples: list[tuple[Path, int]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool | None,
    prefetch_factor: int,
    obs_cfg: ObsConfig,
    seed: int = 0,
    amp_dtype: torch.dtype | None = None,
    loss_cfg: LossConfig = DEFAULT_LOSS_CFG,
    capture: FrameRecordCapture | None = None,
    elim_bin_edges: tuple[int, ...] | None = None,
    elim_head_variant: str | None = None,
) -> dict:
    """Run one full validation pass and return the summary metrics.

    Returns a flat dict suitable for nesting under `val:` in the per-epoch
    JSONL row. Top-1/top-3, pass_acc, pass_frac, and the action histograms
    are `None`-guarded against empty denominators (see module docstring).

    `capture`, when given, records the per-frame stratified columns
    (`eval.dump`) alongside the summary reductions — same forward, no extra
    pass. The caller owns writing the capture out. With it, the dataset
    carries the frame-provenance scalars (`include_frame_info`); when `None`
    (the default), the pass is unchanged.

    DataLoader knobs (`num_workers` / `pin_memory` / `prefetch_factor`)
    are caller-supplied — `run_val` mirrors the train loop's choices for
    apples-to-apples throughput numbers. `pin_memory=None` resolves to
    auto (True iff device is CUDA).
    """
    val_start = time.perf_counter()
    acc = LossAccumulator(loss_cfg)
    ent_meter = PolicyEntropyMeter()
    dist_meter = ActionDistMeter()
    # Elim diagnostics ride the same forward; the time_bin head's ElimMeter is
    # built only for that variant. The next_death head has no in-loop meter yet
    # (its diagnostics are computed offline from the dump's raw columns); its
    # `next_elim` loss still rides through the LossAccumulator below.
    elim_meter = (
        ElimMeter(n_bins=len(elim_bin_edges) + 1, target_tau=loss_cfg.elim_target_tau)
        if elim_bin_edges is not None and elim_head_variant == "time_bin"
        else None
    )
    n_top1_correct = 0
    n_top3_correct = 0
    n_pass_correct = 0
    n_pass_observed = 0

    # Val shuffle is intentionally deterministic across epochs (fixed seed, no
    # `set_epoch`), so the per-epoch val loss numbers are apples-to-apples —
    # variation across epochs reflects model change, not reorder.
    for host_batch, batch, out in iter_val_forward(
        model,
        val_samples,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        obs_cfg=obs_cfg,
        include_frame_info=capture is not None,
        elim_bin_edges=elim_bin_edges,
        elim_head_variant=elim_head_variant,
        seed=seed,
        amp_dtype=amp_dtype,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    ):
        # The loss re-enters autocast (the forward already ran under it inside
        # `iter_val_forward`); the split is a no-op since bc_loss reuses no
        # model weights. `amp_dtype is None` → autocast disabled, fp32 unchanged.
        with torch.amp.autocast(
            device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None,
        ):
            losses = bc_loss(out, batch, loss_cfg)
        B = batch["obs"].shape[0]
        acc.update(losses, batch_size=B)

        # Outside the autocast block, so the capture's `.float()` math runs
        # fp32 (the logits are fp16-derived under AMP — recorded as
        # `forward_dtype` in the dump meta).
        if capture is not None:
            capture.add_batch(host_batch, batch, out)

        mask = batch["mask"]
        action_target = batch["action_target"]
        is_pass = batch["is_pass"]
        pass_logit = out["pass_logit"]
        policy_logits = out["policy_logits"]

        # Top-k indices on the flat masked layout. The k=3 result serves both
        # top-1 (column 0) and top-3 (any-match across the 3 columns) with a
        # single GPU op per batch.
        masked_logits = flatten_policy_logits(policy_logits, mask)
        topk = torch.topk(masked_logits, k=3, dim=1).indices  # [B, 3]

        # Accuracy denominators differ by metric:
        #   top-k    → non-pass frames (policy head's supervised domain).
        #   pass_acc → all frames (pass head's decision domain).
        non_pass = ~is_pass
        top1 = (topk[:, 0] == action_target) & non_pass
        top3 = (action_target.unsqueeze(1) == topk).any(dim=1) & non_pass
        n_top1_correct += int(top1.sum())
        n_top3_correct += int(top3.sum())

        ent_meter.update(masked_logits, non_pass)

        # Pass head threshold: logit > 0 ≡ sigmoid > 0.5.
        pass_pred = pass_logit > 0
        n_pass_correct += int((pass_pred == is_pass).sum())
        n_pass_observed += int(is_pass.sum())

        dist_meter.update(topk[:, 0], action_target, non_pass)

        if elim_meter is not None and "elim_logits" in out:
            elim_meter.update(
                out["elim_logits"],
                batch["elim_bin_target"],
                batch["alive_mask"],
            )

    duration_sec = time.perf_counter() - val_start
    s = acc.summary()
    n_non_pass = int(s["n_non_pass"])  # a count; summary() widens it to float | int
    n_samples = s["n_samples"]
    top1_acc = n_top1_correct / n_non_pass if n_non_pass > 0 else None
    top3_acc = n_top3_correct / n_non_pass if n_non_pass > 0 else None
    pass_acc = n_pass_correct / n_samples if n_samples > 0 else None
    pass_frac = n_pass_observed / n_samples if n_samples > 0 else None
    samples_per_sec = n_samples / duration_sec if duration_sec > 0 else 0.0

    summary = {
        "policy": s["policy"],
        "value": s["value"],
        "value_soft": s["value_soft"],
        "pass": s["pass"],
        "total": s["total"],
        "n_non_pass": n_non_pass,
        "n_samples": n_samples,
        "top1": top1_acc,
        "top3": top3_acc,
        "policy_entropy": ent_meter.mean(),
        "pass_acc": pass_acc,
        "pass_frac": pass_frac,
        "duration_sec": round(duration_sec, 3),
        "samples_per_sec": round(samples_per_sec, 2),
        # action_target_dist is constant across epochs — duplicated per
        # row for grep-ability when comparing to action_dist.
        "action_dist": dist_meter.pred_dist(),
        "action_target_dist": dist_meter.target_dist(),
    }
    # Elim head metrics — present only when the head is active, so non-elim val
    # rows are unchanged. `elim_soft` is the health metric (vs `elim_soft_floor`,
    # the soft-marginal floor); `elim` (hard CE) is recorded but not the in-loop
    # check at τ>0 (it pays the softness tax — see ElimMeter).
    if elim_meter is not None:
        summary["elim_soft"] = s.get("elim_soft")
        summary["elim"] = s.get("elim")
        summary["elim_top1"] = elim_meter.top1()
        summary["elim_soft_floor"] = elim_meter.soft_floor()
        summary["elim_pred_entropy"] = elim_meter.pred_entropy()
    # who-dies-next: only the loss is surfaced in-loop (no meter yet); the
    # accuracy / ramp / horizon reads are computed offline from the dump.
    if "next_elim" in s:
        summary["next_elim"] = s.get("next_elim")
    return summary
