"""
Behavioral-cloning loss: policy CE + value CE + pass BCE.

Three supervision signals, combined with fixed weights:

    total = policy_ce + λ · value_ce + μ · pass_bce        (λ=0.5, μ=1.0)

Each component is mean-reduced over its eligible samples in the batch:
  - policy_ce is mean over *non-pass* frames only (pass frames carry no
    action signal — they're voluntary "do nothing" moves).
  - value_ce and pass_bce are mean over the full batch.

The mean-per-component reduction means the weights λ, μ control the
*relative* head importance independent of how many pass frames the batch
happens to contain. Per the plan, initial weights are inherited starting
points; revisit if any one head dominates the loss curve.

Layout coupling: the policy head produces NCHW `[B, 8, H, W]`. The action
target is in the cell-major flat layout (`flat_idx = cell_padded * 8 + sub`,
sub = dir·2+split). Per the cross-cutting permute contract (5.18-3 session
note), this module owns the `permute(0,2,3,1).reshape(B, -1)` transform
that lines the two up.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from bc.actions import _PASS_FLAT_IDX


# Loss head weights. Tentative v1 spike defaults from
# `5.17-4-training-spike-plan.md` §3 step 10. Revisit if one head dominates.
LAMBDA_VALUE = 0.5
MU_PASS = 1.0

# Mask logits set to this value before the softmax. Large-negative
# rather than `-inf` to avoid NaN propagation if a downstream op
# touches a masked position. Magnitude is bounded by FP16's representable
# range (max abs ~65504); -1e4 is comfortably inside that and still
# underflows to ~0 after softmax (`e^{-1e4} ≈ 0` in any precision).
# Earlier versions used -1e9 which silently overflowed under AMP.
MASK_NEG = -1e4


def flatten_policy_logits(
    policy_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Flatten NCHW policy logits to cell-major `[B, H·W·8]` and apply
    the legality mask. Owner of the cell-major permute contract (5.18-3
    session note); callers use the flat layout for CE / argmax.
    """
    B = policy_logits.shape[0]
    # Permute moves the direction channel last, matching the action
    # target's cell-major layout (flat_idx = cell_padded*8 + sub,
    # sub = dir*2 + split):
    #   [B, 8, H, W]  →  [B, H, W, 8]  →  [B, H·W·8]
    flat = policy_logits.permute(0, 2, 3, 1).contiguous().reshape(B, -1)
    # Apply legality mask. Illegal positions get MASK_NEG (large-negative,
    # mixed-precision-safe vs -inf) — post-softmax, prob ≈ 0.
    return flat.masked_fill(~mask.reshape(B, -1), MASK_NEG)


def bc_loss(
    model_out: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    One-step loss over a batch.

    Inputs
    ------
    model_out
        - `policy_logits`: `[B, 8, H, W]` (NCHW; from `PolicyHead`)
        - `pass_logit`:    `[B]`         (pre-sigmoid)
        - `value_logits`:  `[B, 8]`

    targets (collated from `dataset.encode_frame`)
        - `mask`:          `[B, H, W, 8]` bool, per-cell legality
        - `action_target`: `[B]` int64; flat cell-major index or `-1` for pass
        - `is_pass`:       `[B]` bool
        - `value_target`:  `[B]` int64 (placement class 0..7)

    Returns a dict
        - `total`:    scalar loss for `.backward()`
        - `policy`:   policy CE (mean over non-pass frames; 0 if batch is all pass)
        - `value`:    value CE (mean over batch)
        - `pass`:     pass BCE (mean over batch)
        - `n_non_pass`: int scalar, number of non-pass frames in the batch
                       (debugging signal for the overfit harness)
    """
    policy_logits = model_out["policy_logits"]  # [B, 8, H, W]
    pass_logit = model_out["pass_logit"]        # [B]
    value_logits = model_out["value_logits"]    # [B, 8]

    mask = targets["mask"]                      # [B, H, W, 8] bool
    action_target = targets["action_target"]    # [B] int64
    is_pass = targets["is_pass"]                # [B] bool
    value_target = targets["value_target"]      # [B] int64

    _B = policy_logits.shape[0]

    # --- Policy CE ---
    # F.cross_entropy applies log-softmax internally; the flatten helper's
    # MASK_NEG fill is equivalent to a multiplicative mask on the
    # probability simplex (masked positions → prob ≈ 0).
    policy_logits_masked = flatten_policy_logits(policy_logits, mask)

    # Cross-entropy with ignore_index=-1 to skip pass frames.
    # Pass frames have action_target == -1 by construction (see
    # `bc/actions.py` `_PASS_FLAT_IDX`). They contribute no policy gradient.
    n_non_pass = int((~is_pass).sum().item())
    if n_non_pass > 0:
        policy_ce = F.cross_entropy(
            policy_logits_masked,
            action_target,
            ignore_index=_PASS_FLAT_IDX,
            reduction="mean",
        )
    else:
        # All-pass batch: F.cross_entropy with all targets ignored returns
        # NaN (mean over empty set). Defensive guard. Won't fire in the
        # fixed overfit batch but real corpus batches occasionally hit it.
        policy_ce = policy_logits_masked.sum() * 0.0  # zero with gradient path

    # --- Value CE ---
    value_ce = F.cross_entropy(value_logits, value_target, reduction="mean")

    # --- Pass BCE ---
    # `pass_logit` is pre-sigmoid; the `_with_logits` variant fuses sigmoid
    # + BCE for numerical stability. Target needs to be float for BCE.
    pass_bce = F.binary_cross_entropy_with_logits(
        pass_logit, is_pass.float(), reduction="mean"
    )

    total = policy_ce + LAMBDA_VALUE * value_ce + MU_PASS * pass_bce

    return {
        "total": total,
        "policy": policy_ce,
        "value": value_ce,
        "pass": pass_bce,
        "n_non_pass": torch.tensor(n_non_pass),
    }


@dataclass
class LossAccumulator:
    """
    Sample-weighted running means over `bc_loss` returns for epoch summaries.

    Aggregation rules (see module docstring for the per-batch definitions):
      - `policy` is mean-over-non-pass-frames per batch, so the epoch mean
        weights each batch by its `n_non_pass`. An all-pass batch's policy
        loss is 0 by the `bc_loss` defensive guard, but more importantly
        its `n_non_pass` is 0 → it contributes zero weight and zero sum,
        which is the right thing.
      - `value`, `pass` are mean-over-full-batch per batch, so the epoch
        mean weights each batch by its sample count `B`.
      - `total` is *derived* from the component epoch means using the
        fixed weights `LAMBDA_VALUE`, `MU_PASS`. This preserves the
        identity `total == policy + λ·value + μ·pass` at every
        aggregation level — useful when reading the log to see which
        head is driving the loss.

    One accumulator per epoch per split (train + val each get their own).
    No `reset()` — instantiate a fresh accumulator at the start of each
    epoch / val pass.
    """

    n_non_pass: int = 0
    n_samples: int = 0
    sum_policy: float = 0.0  # weighted by n_non_pass per batch
    sum_value: float = 0.0   # weighted by batch_size per batch
    sum_pass: float = 0.0    # weighted by batch_size per batch

    def update(
        self,
        losses: dict[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        """Fold one batch's `bc_loss` return dict into the running totals."""
        n_np = int(losses["n_non_pass"].item())
        self.n_non_pass += n_np
        self.n_samples += batch_size
        self.sum_policy += losses["policy"].item() * n_np
        self.sum_value += losses["value"].item() * batch_size
        self.sum_pass += losses["pass"].item() * batch_size

    def summary(self) -> dict[str, float | int]:
        """
        Epoch-level loss summary as plain floats.

        Pre-update accumulator returns zeros (no div-by-zero). An epoch
        consisting entirely of all-pass batches also returns 0 for the
        policy mean — `n_non_pass` is 0 across all batches.
        """
        policy = self.sum_policy / self.n_non_pass if self.n_non_pass > 0 else 0.0
        value = self.sum_value / self.n_samples if self.n_samples > 0 else 0.0
        pass_ = self.sum_pass / self.n_samples if self.n_samples > 0 else 0.0
        total = policy + LAMBDA_VALUE * value + MU_PASS * pass_
        return {
            "policy": policy,
            "value": value,
            "pass": pass_,
            "total": total,
            "n_non_pass": self.n_non_pass,
            "n_samples": self.n_samples,
        }
