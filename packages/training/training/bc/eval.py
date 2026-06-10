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

    - **`action_dist` / `action_target_dist`**: 8-bucket histograms over
      `(direction, split)` sub-actions — directional bias / mode collapse
      alarm. `action_target_dist` is constant across epochs (it's a
      property of the val set) but echoed in every row for grep-ability
      when comparing the two.

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
from torch.utils.data import DataLoader

from training.bc.dataset import IterableDataset, assert_safe_loader
from training.bc.loss import (
    DEFAULT_LOSS_CFG,
    LossAccumulator,
    LossConfig,
    bc_loss,
    flatten_policy_logits,
)
from training.bc.obs_config import ObsConfig
from training.shared.device import dataloader_kwargs, move_batch, obs_for_model


# 8-bucket action histogram keys. Index = `flat_action_idx % 8` =
# `dir * 2 + split` (see `bc/actions.py`). N=0, E=1, S=2, W=3.
_SUB_NAMES = (
    "n_move", "n_split",
    "e_move", "e_split",
    "s_move", "s_split",
    "w_move", "w_split",
)


def _dist(counts: torch.Tensor, n: int) -> dict[str, float | None]:
    """Bucket counts → named-fraction dict, or all-None if `n == 0`."""
    if n == 0:
        return {name: None for name in _SUB_NAMES}
    return {name: float(counts[i].item() / n) for i, name in enumerate(_SUB_NAMES)}


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
) -> dict:
    """Run one full validation pass and return the summary metrics.

    Returns a flat dict suitable for nesting under `val:` in the per-epoch
    JSONL row. Top-1/top-3, pass_acc, pass_frac, and the action histograms
    are `None`-guarded against empty denominators (see module docstring).

    DataLoader knobs (`num_workers` / `pin_memory` / `prefetch_factor`)
    are caller-supplied — `run_val` mirrors the train loop's choices for
    apples-to-apples throughput numbers. `pin_memory=None` resolves to
    auto (True iff device is CUDA).
    """
    # Val shuffle is intentionally deterministic across epochs (we never
    # call `set_epoch`), so the per-epoch val loss numbers are apples-to-
    # apples — variation across epochs reflects model change, not reorder.
    val_ds = IterableDataset(samples=val_samples, seed=seed, obs_cfg=obs_cfg)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        **dataloader_kwargs(
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            device=device,
        ),
    )
    assert_safe_loader(val_loader)

    val_start = time.perf_counter()
    acc = LossAccumulator(loss_cfg)
    n_top1_correct = 0
    n_top3_correct = 0
    n_pass_correct = 0
    n_pass_observed = 0
    pred_counts = torch.zeros(8, dtype=torch.long)
    target_counts = torch.zeros(8, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch(batch, device)
            # Same autocast logic as the train loop, minus the GradScaler
            # (no backward in val). `amp_dtype is None` → autocast is a
            # no-op via `enabled=False`, so fp32 callers see no change.
            with torch.amp.autocast(
                device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None,
            ):
                out = model(obs_for_model(batch, amp_dtype), batch["valid_mask"])
                losses = bc_loss(out, batch, loss_cfg)
            B = batch["obs"].shape[0]
            acc.update(losses, batch_size=B)

            mask = batch["mask"]
            action_target = batch["action_target"]
            is_pass = batch["is_pass"]
            pass_logit = out["pass_logit"]
            policy_logits = out["policy_logits"]

            # Top-k indices on the flat masked layout. The k=3 result
            # serves both top-1 (column 0) and top-3 (any-match across
            # the 3 columns) with a single GPU op per batch.
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

            # Pass head threshold: logit > 0 ≡ sigmoid > 0.5.
            pass_pred = pass_logit > 0
            n_pass_correct += int((pass_pred == is_pass).sum())
            n_pass_observed += int(is_pass.sum())

            # 8-bucket histograms restricted to non-pass frames so
            # `action_target % 8` is well-defined (pass frames have -1).
            # Move to CPU before bincount — keeps the tiny per-batch
            # bookkeeping device-agnostic.
            if non_pass.any():
                # ===================================================================
                # TODO(mps-val-crash): BROKEN ON MPS — local val-on runs crash here.
                #
                # A bool-mask index into an int column on the MPS backend
                # (`topk[:, 0][non_pass]`) returns garbage indices, raising:
                #   AcceleratorError: index <huge-int> out of bounds ... size 64
                #
                # Reproduced 2026-05-28 on M1 Max. The earlier "split the index
                # into two ops" dodge (what the code below already does) no longer
                # dodges it — the bool-mask gather itself is the failing op. A POC
                # on 2026-05-21 ran val cleanly, so it regressed sometime after.
                #
                # Impact: blocks every local MPS val pass (run_val). Cloud/CUDA is
                # unaffected, so cloud training + val still works — this only bites
                # local smoke runs that don't pass --skip-val.
                #
                # Likely fix: move to CPU BEFORE the bool-mask gather, e.g.
                #   npm = non_pass.cpu()
                #   pred_subs   = (topk[:, 0].cpu()[npm]) % 8
                #   target_subs = (action_target.cpu()[npm]) % 8
                # Deferred — out of scope for the train.py refactor series; wants a
                # focused pass that verifies the fix against an MPS val run.
                # ===================================================================
                pred_subs = (topk[:, 0][non_pass] % 8).cpu()
                target_subs = (action_target[non_pass] % 8).cpu()
                pred_counts += torch.bincount(pred_subs, minlength=8)
                target_counts += torch.bincount(target_subs, minlength=8)

    duration_sec = time.perf_counter() - val_start
    s = acc.summary()
    n_non_pass = int(s["n_non_pass"])  # a count; summary() widens it to float | int
    n_samples = s["n_samples"]
    top1_acc = n_top1_correct / n_non_pass if n_non_pass > 0 else None
    top3_acc = n_top3_correct / n_non_pass if n_non_pass > 0 else None
    pass_acc = n_pass_correct / n_samples if n_samples > 0 else None
    pass_frac = n_pass_observed / n_samples if n_samples > 0 else None
    samples_per_sec = n_samples / duration_sec if duration_sec > 0 else 0.0

    return {
        "policy": s["policy"],
        "value": s["value"],
        "value_soft": s["value_soft"],
        "pass": s["pass"],
        "total": s["total"],
        "n_non_pass": n_non_pass,
        "n_samples": n_samples,
        "top1": top1_acc,
        "top3": top3_acc,
        "pass_acc": pass_acc,
        "pass_frac": pass_frac,
        "duration_sec": round(duration_sec, 3),
        "samples_per_sec": round(samples_per_sec, 2),
        # action_target_dist is constant across epochs — duplicated per
        # row for grep-ability when comparing to action_dist.
        "action_dist": _dist(pred_counts, n_non_pass),
        "action_target_dist": _dist(target_counts, n_non_pass),
    }
