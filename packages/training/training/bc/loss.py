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

from dataclasses import dataclass, field, fields
from typing import Any
import warnings

import torch
import torch.nn.functional as F

from training.bc.actions import _PASS_FLAT_IDX
from training.bc.aux_heads import AuxHeadSpec
from training.bc.model import ModelOut, flatten_policy_logits
from training.bc.soft_target import _soft_target_kernel


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

    @classmethod
    def validate_partial(cls, d: dict) -> list[str]:
        """Pre-flight check of a config-file `loss:` block: flag unknown knobs.
        Mirrors `ObsConfig.validate_partial` — value ranges are enforced by
        `__post_init__` at construction; this only catches typo'd field names."""
        valid = {f.name for f in fields(cls)}
        return [f"unknown LossConfig field: {k!r}" for k in d if k not in valid]


# Shared no-knobs-set instance: the default for `bc_loss` / `run_val`
# callers outside the configured train loop (scripts, tests).
DEFAULT_LOSS_CFG = LossConfig()


def bc_loss(
    model_out: ModelOut,
    targets: dict[str, torch.Tensor],
    cfg: LossConfig = DEFAULT_LOSS_CFG,
) -> dict[str, torch.Tensor]:
    """
    One-step loss over a batch.

    Inputs
    ------
    model_out (a `ModelOut`)
        - `policy_logits`: `[B, 8, H, W]` (NCHW; from `PolicyHead`)
        - `pass_logit`:    `[B]`         (pre-sigmoid)
        - `value_logits`:  `[B, 8]`
        - `aux`:               active aux-head logits, keyed by `output_key`
        - `active_aux_specs`:  the built aux heads — the loss computes a term for each

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

    The loss computes a term for each spec in `model_out.active_aux_specs`.
    `ModelOut` asserts that each active head has its expected output.

    When `time_bin` is among `model_out.active_aux_specs`, three more keys appear
    (absent otherwise, so non-elim runs are unchanged):
        - `elim`:      hard elim CE, masked-mean over alive (player, frame)
                       pairs — the reporting metric
        - `elim_soft`: soft-target elim CE, the trained objective
                       (same tensor as `elim` at τ=0)
        - `n_elim`:    0-d int tensor, count of alive (player, frame) pairs
                       (the accumulator weight for the elim means)

    When `next_death` is among `model_out.active_aux_specs` (mutually exclusive
    with `time_bin`), three keys appear instead:
        - `next_elim`:      hard who-is-removed-next CE against the one-hot next
                            victim, mean over frames with a defined next removal
                            (winner-tail frames excluded via -1) — the reporting
                            metric
        - `next_elim_soft`: soft-target CE, the trained objective (same tensor as
                            `next_elim` at τ=0)
        - `n_next_elim`:    0-d int tensor, count of those frames (the accumulator
                            weight for the next_elim means)
    """
    policy_logits = model_out.policy_logits  # [B, 8, H, W]
    pass_logit = model_out.pass_logit        # [B]
    value_logits = model_out.value_logits    # [B, 8]

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

    out = {
        "policy": policy_ce,
        "value": value_ce,
        "value_soft": value_soft,
        "pass": pass_bce,
        "n_non_pass": n_non_pass,
    }

    # Aux-head terms: each active head computes its loss and metrics.
    # `_assemble_total` then folds the trained terms into `total`.
    for spec in model_out.active_aux_specs:
        res = spec.loss(model_out, targets, cfg)
        out.update(res.metrics)
        out[spec.count_key] = res.count

    out["total"] = _assemble_total(out, cfg, model_out.active_aux_specs)
    return out


@dataclass
class _AuxAccum:
    """Running sums for one aux spec's metrics, weighted by its own count.

    `count` and `sums` move together — both fold in the same per-batch denominator
    (the head's count, e.g. alive-pair count) — so they live in one object rather
    than parallel dicts that have to be kept in sync.
    """

    count: int = 0
    sums: dict[str, float] = field(default_factory=dict)  # metric_key → Σ value·count

    def add(
        self,
        n: int,
        metric_keys: tuple[str, ...],
        losses: dict[str, torch.Tensor],
    ) -> None:
        self.count += n
        for k in metric_keys:
            self.sums[k] = self.sums.get(k, 0.0) + losses[k].item() * n


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
      - Each aux head's metrics weight by the head's *own* count (`spec.count_key`
        — alive-pair count for time_bin, defined-next-victim count for next_death),
        not n_samples. The active heads come from `active_aux_specs` (the model's
        own list), so the fold iterates exactly the built heads.
      - `total` is built by the shared `_assemble_total` (same call as `bc_loss`),
        so the identity holds at every aggregation level. The *soft* term is the
        trained objective (== hard at τ=0). `cfg` and `active_aux_specs` must
        match those given to `bc_loss`.

    One accumulator per epoch per split (train + val each get their own).
    No `reset()` — instantiate a fresh accumulator at the start of each
    epoch / val pass.
    """

    cfg: LossConfig = field(default_factory=LossConfig)
    active_aux_specs: tuple[AuxHeadSpec, ...] = ()
    n_non_pass: int = 0
    n_samples: int = 0
    sum_policy: float = 0.0      # weighted by n_non_pass per batch
    sum_value: float = 0.0       # weighted by batch_size per batch
    sum_value_soft: float = 0.0  # weighted by batch_size per batch
    sum_pass: float = 0.0        # weighted by batch_size per batch
    # Per-aux-spec running sums, keyed by spec.name; one entry per active head,
    # built in __post_init__ from `active_aux_specs`. Empty for non-aux runs.
    aux: dict[str, _AuxAccum] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # One accumulator per active head, built up front from the declared specs
        # rather than lazily on first sighting of a batch key — so `update` and
        # `summary` index directly without a presence check.
        self.aux = {spec.name: _AuxAccum() for spec in self.active_aux_specs}

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
        # Each active head weights its metrics by the head's own count.
        for spec in self.active_aux_specs:
            n = int(losses[spec.count_key].item())
            self.aux[spec.name].add(n, spec.metric_keys, losses)

    def summary(self) -> dict[str, float | int]:
        """
        Epoch-level loss summary as plain floats.

        Pre-update accumulator returns zeros (no div-by-zero). An epoch
        consisting entirely of all-pass batches also returns 0 for the
        policy mean — `n_non_pass` is 0 across all batches.
        """

        # TODO: Should we do something other other than return 0 if divisor is 0 here?
        policy = _divide_safe(self.sum_policy, self.n_non_pass)
        value = _divide_safe(self.sum_value, self.n_samples)
        value_soft = _divide_safe(self.sum_value_soft, self.n_samples)
        pass_ = _divide_safe(self.sum_pass, self.n_samples)

        summary: dict[str, float | int] = {
            "policy": policy,
            "value": value,
            "value_soft": value_soft,
            "pass": pass_,
            "n_non_pass": self.n_non_pass,
            "n_samples": self.n_samples,
        }
        # Reconstruct each active head's mean metrics; collect the ones that
        # contributed so `_assemble_total` folds exactly those into `total`.
        contributing: list[AuxHeadSpec] = []
        for spec in self.active_aux_specs:
            a = self.aux[spec.name]
            if a.count <= 0:
                # In a real training run, `count` being 0 is probably almost always bug.
                # But it could happen in a "smoke run", so we warn instead of throwing an error.
                # Throwing an error in the middle of a run should be done very cautiously.
                warnings.warn(
                    f"aux head {spec.name!r} saw zero countable frames "
                    f"({spec.count_key}=0) across the epoch; skipping its metrics"
                )
                continue
            for k in spec.metric_keys:
                summary[k] = a.sums[k] / a.count
            summary[spec.count_key] = a.count
            contributing.append(spec)
        summary["total"] = _assemble_total(summary, self.cfg, tuple(contributing))
        return summary


def _assemble_total(
    metrics: dict[str, Any],
    cfg: LossConfig,
    specs: tuple[AuxHeadSpec, ...],
) -> Any:
    """The one definition of how `total` is composed from component metrics — shared
    by `bc_loss` (live tensors) and `LossAccumulator.summary` (epoch-mean floats), so
    the per-batch and epoch totals can't drift. Reads `policy`/`value_soft`/`pass`
    plus each spec's `term_key` from `metrics` (the caller must have populated them);
    polymorphic over tensor/float via the shared `+`/`*`."""
    total = (
        metrics["policy"]
        + cfg.lambda_value * metrics["value_soft"]
        + cfg.mu_pass * metrics["pass"]
    )
    for spec in specs:
        total = total + getattr(cfg, spec.weight_attr) * metrics[spec.term_key]
    return total


def _divide_safe(val: float, divisor: int) -> float:
    """Return 0.0 if divisor is zero"""
    return val / divisor if divisor > 0 else 0.0
