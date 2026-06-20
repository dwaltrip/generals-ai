"""
Behavioral-cloning loss: policy CE + value CE + pass BCE, plus an optional
next-elimination aux term.

Supervision signals, combined per `LossConfig` weights:

    total = policy_ce + λ · value_soft + μ · pass_bce  [+ λ_elim · elim_soft]

The bracketed elim term is present only when the model emits `elim_logits` (the
next-elimination head is built); otherwise the loss is exactly the three-signal
form above.

Each component is mean-reduced over its eligible samples in the batch:
  - policy_ce is mean over *non-pass* frames only (pass frames carry no
    action signal — they're voluntary "do nothing" moves).
  - value losses and pass_bce are mean over the full batch.
  - elim CE is mean over alive (player, frame) pairs (the per-player masked
    mean — real, currently-alive players only).

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
    # Next-elimination aux-head knobs (6.13-5). `lambda_elim` weights the elim
    # term in the total (0 = no elim gradient); `elim_target_tau` is its soft-
    # ordinal smoothing (tau=0 is one-hot, same family as `value_target_tau`).
    # `elim_bin_weights` is optional per-bin CE weights for the class imbalance
    # measured in 6.13-6 (None = unweighted; Stage 1 runs unweighted, weights
    # are the pre-registered floor-miss remedy). When set, the weight applies to
    # both the hard reporting CE and the soft objective, so `elim == elim_soft`
    # still holds at tau=0. The elim term is a no-op unless the model emits
    # `elim_logits`, so these defaults leave non-elim runs untouched.
    lambda_elim: float = 0.0
    elim_target_tau: float = 0.0
    elim_bin_weights: tuple[float, ...] | None = None
    # next_death (who-is-removed-next) soft target. τ>0 replaces the one-hot
    # next-victim label with a distribution over the *present* players decaying
    # with how much later each is removed than the next victim:
    # `p_i ∝ exp(-(removal_dt_i - removal_dt_min)/τ)`, τ in ticks. Unlike the
    # ordinal `elim_target_tau` (time_bin bins), this is a per-frame data-dependent
    # distribution over nominal player channels, built in the loss from the
    # per-channel `next_elim_removal_dt`. τ=0 keeps the hard label (current
    # behavior). The near-tie relief auto-adapts to game phase (crowded boards
    # have tight removal gaps → softer; sparse late boards → near one-hot).
    next_elim_target_tau: float = 0.0

    def __post_init__(self) -> None:
        if self.lambda_value < 0:
            raise ValueError(f"lambda_value must be >= 0; got {self.lambda_value}")
        if self.mu_pass < 0:
            raise ValueError(f"mu_pass must be >= 0; got {self.mu_pass}")
        if self.value_target_tau < 0:
            raise ValueError(
                f"value_target_tau must be >= 0; got {self.value_target_tau}"
            )
        if self.lambda_elim < 0:
            raise ValueError(f"lambda_elim must be >= 0; got {self.lambda_elim}")
        if self.elim_target_tau < 0:
            raise ValueError(
                f"elim_target_tau must be >= 0; got {self.elim_target_tau}"
            )
        if self.next_elim_target_tau < 0:
            raise ValueError(
                f"next_elim_target_tau must be >= 0; got {self.next_elim_target_tau}"
            )
        if self.elim_bin_weights is not None:
            weights = tuple(float(w) for w in self.elim_bin_weights)
            if any(w < 0 for w in weights):
                raise ValueError(
                    f"elim_bin_weights must be non-negative; got {weights}"
                )
            # Coerce to tuple (a config JSON hands a list) so the frozen
            # dataclass stays hashable and the @cache weight-tensor key is stable.
            object.__setattr__(self, "elim_bin_weights", weights)


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


@cache
def _elim_weight_tensor(
    weights: tuple[float, ...], device: torch.device
) -> torch.Tensor:
    """Per-bin CE weight vector for the elim head, as an fp32 device tensor.

    Cached per `(weights, device)` — the tuple is hashable (LossConfig coerces
    it) so it keys the cache directly. fp32 because autocast routes
    cross_entropy to fp32.
    """
    return torch.tensor(weights, dtype=torch.float32, device=device)


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

    When `model_out` carries `elim_logits` (the time_bin elim head is built),
    three more keys appear (absent otherwise, so non-elim runs are unchanged):
        - `elim`:      hard elim CE, masked-mean over alive (player, frame)
                       pairs — the reporting metric
        - `elim_soft`: soft-target elim CE, the trained objective
                       (same tensor as `elim` at τ=0)
        - `n_elim`:    0-d int tensor, count of alive (player, frame) pairs
                       (the accumulator weight for the elim means)

    When `model_out` carries `next_elim_logits` (the next_death elim head is
    built — mutually exclusive with `elim_logits`), three keys appear instead:
        - `next_elim`:      hard who-is-removed-next CE against the one-hot next
                            victim, mean over frames with a defined next removal
                            (winner-tail frames excluded via -1) — the reporting
                            metric
        - `next_elim_soft`: soft-target CE, the trained objective (same tensor as
                            `next_elim` at τ=0)
        - `n_next_elim`:    0-d int tensor, count of those frames (the accumulator
                            weight for the next_elim means)
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

    out = {
        "policy": policy_ce,
        "value": value_ce,
        "value_soft": value_soft,
        "pass": pass_bce,
        "n_non_pass": n_non_pass,
    }

    # --- Elim CE (only when the model emits the head's logits) ---
    # A disabled head is a clean no-op: existing consumers are untouched, and
    # the per-player masked-mean reduction added here is the discrimination the
    # value head lacks.
    if "elim_logits" in model_out:
        elim_logits = model_out["elim_logits"]          # [B, 8, n_bins]
        elim_bin_target = targets["elim_bin_target"]    # [B, 8] int64
        alive_mask = targets["alive_mask"]         # [B, 8] bool
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

        total = total + cfg.lambda_elim * elim_soft
        out["elim"] = elim
        out["elim_soft"] = elim_soft
        out["n_elim"] = n_elim

    # --- Who-dies-next CE (only when the next_death head emits its logits) ---
    # The `next_death` elim variant: a single cross-player softmax per frame over
    # "which present player is removed from the board next". Shares `lambda_elim`
    # with the time_bin head — only one variant is built per model, so they never
    # both contribute. The cross-player softmax is over the *present* domain (the
    # board-removal event's domain), not the alive domain the time_bin head uses.
    if "next_elim_logits" in model_out:
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
        total = total + cfg.lambda_elim * next_elim_soft
        out["next_elim"] = next_elim
        out["next_elim_soft"] = next_elim_soft
        out["n_next_elim"] = n_next_elim

    out["total"] = total
    return out


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
        `total == policy + λ·value_soft + μ·pass [+ λ_elim·elim_soft]` at
        every aggregation level — useful when reading the log to see which
        head is driving the loss. `cfg` must match the one given to `bc_loss`.
        The elim term contributes only when elim batches were folded in
        (otherwise `elim_soft` and `lambda_elim` are both 0).

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
    # Elim means use their own weight: the alive (player, frame) count, distinct
    # from both n_non_pass and n_samples. Stay zero for non-elim runs (the keys
    # are absent from `bc_loss`'s return), so the elim term drops out of total.
    n_elim: int = 0
    sum_elim: float = 0.0        # weighted by n_elim per batch
    sum_elim_soft: float = 0.0   # weighted by n_elim per batch
    # who-dies-next (next_death variant) means: weight by the count of frames
    # with a defined next victim. Zero for non-next_death runs (the keys are
    # absent from `bc_loss`'s return), so the term drops out of total.
    n_next_elim: int = 0
    sum_next_elim: float = 0.0       # weighted by n_next_elim per batch
    sum_next_elim_soft: float = 0.0  # weighted by n_next_elim per batch

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
        # Elim keys appear only when the head is built; weight by n_elim.
        if "elim" in losses:
            n_e = int(losses["n_elim"].item())
            self.n_elim += n_e
            self.sum_elim += losses["elim"].item() * n_e
            self.sum_elim_soft += losses["elim_soft"].item() * n_e
        # next_death keys appear only when that variant is built; weight by the
        # defined-next-victim frame count.
        if "next_elim" in losses:
            n_nx = int(losses["n_next_elim"].item())
            self.n_next_elim += n_nx
            self.sum_next_elim += losses["next_elim"].item() * n_nx
            self.sum_next_elim_soft += losses["next_elim_soft"].item() * n_nx

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
        # Elim means weight by the running alive-pair count, NOT n_samples.
        elim = self.sum_elim / self.n_elim if self.n_elim > 0 else 0.0
        elim_soft = self.sum_elim_soft / self.n_elim if self.n_elim > 0 else 0.0
        # who-dies-next means weight by the running defined-next-victim count.
        next_elim = (
            self.sum_next_elim / self.n_next_elim if self.n_next_elim > 0 else 0.0
        )
        next_elim_soft = (
            self.sum_next_elim_soft / self.n_next_elim if self.n_next_elim > 0 else 0.0
        )
        # Hardcoded total identity — must carry the elim terms or the component
        # sum silently breaks for elim runs. Inert for non-elim runs: lambda_elim
        # defaults to 0, and elim_soft / next_elim_soft are 0 when no such batches
        # were folded. The two elim variants are mutually exclusive, so at most one
        # of elim_soft / next_elim_soft is nonzero. The *soft* terms are the trained
        # objective (== hard at τ=0), matching `bc_loss`'s `total`.
        total = (
            policy
            + self.cfg.lambda_value * value_soft
            + self.cfg.mu_pass * pass_
            + self.cfg.lambda_elim * elim_soft
            + self.cfg.lambda_elim * next_elim_soft
        )
        summary: dict[str, float | int] = {
            "policy": policy,
            "value": value,
            "value_soft": value_soft,
            "pass": pass_,
            "total": total,
            "n_non_pass": self.n_non_pass,
            "n_samples": self.n_samples,
        }
        # Presence-gate the elim entries so non-elim summaries are byte-identical
        # to before.
        if self.n_elim > 0:
            summary["elim"] = elim
            summary["elim_soft"] = elim_soft
            summary["n_elim"] = self.n_elim
        if self.n_next_elim > 0:
            summary["next_elim"] = next_elim
            summary["next_elim_soft"] = next_elim_soft
            summary["n_next_elim"] = self.n_next_elim
        return summary
