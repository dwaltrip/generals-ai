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

import torch
from torch.utils.data import DataLoader

from bc.dataset import IterableDataset
from bc.loss import LossAccumulator, bc_loss, flatten_policy_logits
from shared.device import move_batch


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
    seed: int = 0,
) -> dict:
    """Run one full validation pass and return the summary metrics.

    Returns a flat dict suitable for nesting under `val:` in the per-epoch
    JSONL row. Top-1/top-3, pass_acc, pass_frac, and the action histograms
    are `None`-guarded against empty denominators (see module docstring).
    """
    val_ds = IterableDataset(samples=val_samples, seed=seed)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    acc = LossAccumulator()
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
            out = model(batch["obs"])
            losses = bc_loss(out, batch)
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
                pred_subs = (topk[non_pass, 0] % 8).cpu()
                target_subs = (action_target[non_pass] % 8).cpu()
                pred_counts += torch.bincount(pred_subs, minlength=8)
                target_counts += torch.bincount(target_subs, minlength=8)

    s = acc.summary()
    n_non_pass = s["n_non_pass"]
    n_samples = s["n_samples"]
    top1_acc = n_top1_correct / n_non_pass if n_non_pass > 0 else None
    top3_acc = n_top3_correct / n_non_pass if n_non_pass > 0 else None
    pass_acc = n_pass_correct / n_samples if n_samples > 0 else None
    pass_frac = n_pass_observed / n_samples if n_samples > 0 else None

    return {
        "policy": s["policy"],
        "value": s["value"],
        "pass": s["pass"],
        "total": s["total"],
        "n_non_pass": n_non_pass,
        "n_samples": n_samples,
        "top1": top1_acc,
        "top3": top3_acc,
        "pass_acc": pass_acc,
        "pass_frac": pass_frac,
        # action_target_dist is constant across epochs — duplicated per
        # row for grep-ability when comparing to action_dist.
        "action_dist": _dist(pred_counts, n_non_pass),
        "action_target_dist": _dist(target_counts, n_non_pass),
    }
