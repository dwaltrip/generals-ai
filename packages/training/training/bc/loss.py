"""
Behavioral-cloning loss: policy CE + value CE + pass BCE.

Three supervision signals, combined per `LossConfig` weights:

    total = policy_ce + λ · value_soft + μ · pass_bce

Each component is mean-reduced over its eligible samples in the batch:
  - policy_ce is mean over *non-pass* frames only (pass frames carry no
    action signal — they're voluntary "do nothing" moves).
  - value losses and pass_bce are mean over the full batch.

The mean-per-component reduction means the weights λ, μ control the
*relative* head importance independent of how many pass frames the batch
happens to contain.

The value head has two CE readings, both always computed:
  - `value` — hard CE against the one-hot placement class. The reporting
    metric: comparable across runs and against the placement-entropy floors.
  - `value_soft` — CE against soft ordinal targets (`value_target_tau`),
    the *trained* objective. Equal to `value` at τ=0.

Layout coupling: the policy head produces NCHW `[B, 8, H, W]`; the action
target is in the cell-major flat layout (`flat_idx = cell_padded * 8 + sub`,
sub = dir·2+split). The transform between the two is owned by the model's
output contract (`flatten_policy_logits` in `model/heads/policy.py`); the
policy CE here applies it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

import torch
import torch.nn.functional as F

from training.bc.actions import _PASS_FLAT_IDX
from training.bc.model import flatten_policy_logits


@dataclass(frozen=True)
class LossConfig:
    """Objective knobs, threaded from `TrainConfig` into `bc_loss` /
    `LossAccumulator` as one object so future loss knobs don't re-touch
    every signature on the call chain.

    Defaults reproduce the original fixed-constant behavior exactly
    (λ=0.5 / μ=1.0 from `5.17-4-training-spike-plan.md` §3 step 10,
    one-hot value targets), so `LossConfig()` callers are unchanged.
    """

    # Value-head weight in the total. 0 disables the value gradient entirely
    # (head and trunk both receive none) — the "quasi-control" arm.
    lambda_value: float = 0.5
    # Pass-head weight in the total.
    mu_pass: float = 1.0
    # Soft ordinal placement targets (SORD / HL-Gauss family): the one-hot
    # value target is replaced by a distribution decaying with rank distance,
    # row-softmax of -|k - k*|/τ. Partial credit for near-miss placements
    # densifies the value gradient (coarse "winning vs losing" signal earns
    # loss reduction without nailing the exact rank) and caps the payoff of
    # memorizing exact labels. τ=0 means one-hot (exact current behavior);
    # the mass an adjacent rank gets relative to the peak is exp(-1/τ)
    # (e.g. τ=0.6 → ~0.19).
    value_target_tau: float = 0.0

    def __post_init__(self) -> None:
        if self.lambda_value < 0:
            raise ValueError(f"lambda_value must be >= 0; got {self.lambda_value}")
        if self.mu_pass < 0:
            raise ValueError(f"mu_pass must be >= 0; got {self.mu_pass}")
        if self.value_target_tau < 0:
            raise ValueError(
                f"value_target_tau must be >= 0; got {self.value_target_tau}"
            )


@cache
def _soft_target_kernel(
    tau: float, n_classes: int, device: torch.device
) -> torch.Tensor:
    """The `[n_classes, n_classes]` soft-target matrix for τ > 0: row k* is
    `softmax(-|k - k*|/τ)`. Row-indexing with the hard targets yields the
    per-sample soft distributions. The row softmax renormalizes edge rows
    (rank 1 / rank 8 have one-sided neighborhoods) automatically.

    Cached per (τ, device) — built once, reused every batch. fp32 on
    purpose: autocast routes cross_entropy to fp32, so the targets match.
    """
    ranks = torch.arange(n_classes, dtype=torch.float32, device=device)
    dist = (ranks.unsqueeze(0) - ranks.unsqueeze(1)).abs()
    return F.softmax(-dist / tau, dim=1)


# Shared no-knobs-set instance: the default for `bc_loss` / `run_val`
# callers outside the configured train loop (scripts, tests).
DEFAULT_LOSS_CFG = LossConfig()


def bc_loss(
    model_out: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    cfg: LossConfig = DEFAULT_LOSS_CFG,
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
        - `value`:    hard value CE (mean over batch) — the reporting metric
        - `value_soft`: soft-target value CE, the trained objective
                       (same tensor as `value` at τ=0)
        - `pass`:     pass BCE (mean over batch)
        - `n_non_pass`: 0-d int tensor, number of non-pass frames in the batch
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
    # `reduction="sum"` returns 0 (not NaN) when every target is ignored, so
    # the all-pass edge case folds in via a safe divide — no host sync, no
    # branch. `clamp(min=1)` only matters when the numerator is 0, so it
    # doesn't bias the mean.
    n_non_pass = (~is_pass).sum()  # 0-d tensor, stays on device
    policy_ce = F.cross_entropy(
        policy_logits_masked,
        action_target,
        ignore_index=_PASS_FLAT_IDX,
        reduction="sum",
    ) / n_non_pass.clamp(min=1)

    # --- Value CE ---
    # Hard CE is always computed: it's the cross-run-comparable metric (and
    # what the placement-entropy floors baseline). The soft variant is the
    # objective when τ > 0 — F.cross_entropy accepts probability targets
    # directly, so the soft path is one row-gather plus the same CE call.
    value_ce = F.cross_entropy(value_logits, value_target, reduction="mean")
    if cfg.value_target_tau > 0:
        kernel = _soft_target_kernel(
            cfg.value_target_tau, value_logits.shape[1], value_logits.device
        )
        value_soft = F.cross_entropy(
            value_logits, kernel[value_target], reduction="mean"
        )
    else:
        value_soft = value_ce

    # --- Pass BCE ---
    # `pass_logit` is pre-sigmoid; the `_with_logits` variant fuses sigmoid
    # + BCE for numerical stability. Target needs to be float for BCE.
    pass_bce = F.binary_cross_entropy_with_logits(
        pass_logit, is_pass.float(), reduction="mean"
    )

    total = policy_ce + cfg.lambda_value * value_soft + cfg.mu_pass * pass_bce

    return {
        "total": total,
        "policy": policy_ce,
        "value": value_ce,
        "value_soft": value_soft,
        "pass": pass_bce,
        "n_non_pass": n_non_pass,
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
      - `value`, `value_soft`, `pass` are mean-over-full-batch per batch, so
        the epoch mean weights each batch by its sample count `B`.
      - `total` is *derived* from the component epoch means using the
        `cfg` weights. This preserves the identity
        `total == policy + λ·value_soft + μ·pass` at every aggregation
        level — useful when reading the log to see which head is driving
        the loss. `cfg` must match the one given to `bc_loss`.

    One accumulator per epoch per split (train + val each get their own).
    No `reset()` — instantiate a fresh accumulator at the start of each
    epoch / val pass.
    """

    cfg: LossConfig = field(default_factory=LossConfig)
    n_non_pass: int = 0
    n_samples: int = 0
    sum_policy: float = 0.0      # weighted by n_non_pass per batch
    sum_value: float = 0.0       # weighted by batch_size per batch
    sum_value_soft: float = 0.0  # weighted by batch_size per batch
    sum_pass: float = 0.0        # weighted by batch_size per batch

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
        self.sum_value_soft += losses["value_soft"].item() * batch_size
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
        value_soft = self.sum_value_soft / self.n_samples if self.n_samples > 0 else 0.0
        pass_ = self.sum_pass / self.n_samples if self.n_samples > 0 else 0.0
        total = policy + self.cfg.lambda_value * value_soft + self.cfg.mu_pass * pass_
        return {
            "policy": policy,
            "value": value,
            "value_soft": value_soft,
            "pass": pass_,
            "total": total,
            "n_non_pass": self.n_non_pass,
            "n_samples": self.n_samples,
        }
