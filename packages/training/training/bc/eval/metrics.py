"""Diagnostic meters that ride the validation pass.

A meter is a self-contained accumulator with a common shape: constructed
before the val loop, fed per-batch tensors via `update(...)`, read once at
the end into the val summary dict. The orchestrator (`eval.run`) owns which
meters get fed; the diagnostic math lives here.

Distinct from `loss.LossAccumulator`, which is objective-coupled — it must
mirror `bc_loss`'s component weights. Meters carry no loss semantics; they
exist to put model-behavior signals (sharpness, collapse, calibration) on
the per-epoch record.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from training.bc.soft_target import _soft_target_kernel


class PolicyEntropyMeter:
    """Running mean of per-frame policy entropy over non-pass frames.

    Per-frame entropy is the Shannon entropy (nats) of the masked policy
    softmax; illegal positions carry `MASK_NEG` logits, so their p·log p
    terms underflow to exactly 0 — no NaN guard needed.

    Restricted to non-pass frames via the `non_pass` mask, matching the
    denominator of policy CE and top-k (the policy head's supervision
    domain). One meter per val pass; no reset.
    """

    def __init__(self) -> None:
        self._sum = 0.0
        self._n = 0

    def update(self, masked_logits: torch.Tensor, non_pass: torch.Tensor) -> None:
        """Fold one batch in. `masked_logits` is the `[B, H·W·8]` output of
        `flatten_policy_logits`; `non_pass` is a `[B]` bool mask."""
        # `.float()`: under fp16 autocast the model emits fp16 logits, and
        # entropy sums ~32k tiny p·log p terms — keep the reduction in fp32.
        logp = F.log_softmax(masked_logits.float(), dim=1)
        ent = -(logp.exp() * logp).sum(dim=1)
        # Multiplicative mask: zeroes pass-frame entropies in one fused op,
        # no data-dependent shapes.
        self._sum += float((ent * non_pass).sum().item())
        self._n += int(non_pass.sum())

    def mean(self) -> float | None:
        """Mean entropy in nats; `None` if no non-pass frames were seen."""
        return self._sum / self._n if self._n > 0 else None


class ElimMeter:
    """Next-elimination head diagnostics over alive (player, frame) entries.

    Three reads on the per-epoch record:
      - **top-1 bin accuracy** — argmax bin == target bin.
      - **prediction entropy** — mean Shannon entropy (nats) of the per-player
        bin softmax; the collapse alarm (falling toward 0 while top-1 stalls).
      - **soft-target marginal floor** — the head's trivial baseline, the CE of
        always predicting the *marginal* soft-target distribution. The in-loop
        health check is `elim_soft < floor`: a head that's learned anything
        beats the best constant predictor. The **soft** floor (not a hard-CE
        floor) is the self-consistent comparison for `elim_soft` — at τ>0 a head
        that knows the task still pays a softness tax in hard CE, and that tax is
        baked into this floor too (it's `H(q)` of the smoothed marginal `q`), so
        the comparison is apples-to-apples. See `6.13-5` (metrics) / 6.12-4.

    Built only when the elim head is active; one meter per val pass, no reset.
    Restricted to the alive mask — the loss's supervised domain.
    """

    def __init__(self, n_bins: int, target_tau: float) -> None:
        self.n_bins = n_bins
        self.tau = target_tau
        self._hist = torch.zeros(n_bins, dtype=torch.long)  # marginal of hard targets
        self._ent_sum = 0.0
        self._top1 = 0
        self._n = 0

    def update(
        self,
        elim_logits: torch.Tensor,
        bin_target: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> None:
        """Fold one batch in. `elim_logits` [B, 8, n_bins]; `bin_target` [B, 8]
        int64; `alive_mask` [B, 8] bool. Only alive (player, frame) entries
        count — the same population the elim loss reduces over."""
        if not bool(alive_mask.any()):
            return
        logits = elim_logits[alive_mask]               # [N, n_bins]
        targets = bin_target[alive_mask]               # [N]

        self._top1 += int((logits.argmax(dim=1) == targets).sum())
        # fp32 entropy: under AMP the logits are fp16, and entropy sums tiny
        # p·log p terms — keep the reduction in fp32 like PolicyEntropyMeter.
        logp = F.log_softmax(logits.float(), dim=1)
        self._ent_sum += float((-(logp.exp() * logp).sum(dim=1)).sum().item())
        self._hist += torch.bincount(targets.cpu(), minlength=self.n_bins)
        self._n += int(alive_mask.sum())

    def top1(self) -> float | None:
        return self._top1 / self._n if self._n > 0 else None

    def pred_entropy(self) -> float | None:
        """Mean per-player bin-softmax entropy in nats; `None` if no entries."""
        return self._ent_sum / self._n if self._n > 0 else None

    def soft_floor(self) -> float | None:
        """CE of predicting the marginal soft target — `H(q)`, where `q` is the
        (τ-smoothed) marginal over the alive targets seen. The constant-predictor
        baseline `elim_soft` must beat. `None` if no entries."""
        if self._n == 0:
            return None
        p = self._hist.float() / self._hist.sum()      # hard marginal
        if self.tau > 0:
            kernel = _soft_target_kernel(self.tau, self.n_bins, torch.device("cpu"))
            q = p @ kernel                             # smoothed marginal
        else:
            q = p
        q = q.clamp_min(1e-12)
        return float(-(q * q.log()).sum().item())


# 8-bucket action histogram keys. Index = `flat_action_idx % 8` =
# `dir * 2 + split` (see `bc/actions.py`). N=0, E=1, S=2, W=3.
_SUB_NAMES = (
    "n_move", "n_split",
    "e_move", "e_split",
    "s_move", "s_split",
    "w_move", "w_split",
)


class ActionDistMeter:
    """8-bucket `(direction, split)` histograms of predicted vs demonstrated
    sub-actions over non-pass frames — the directional-bias / mode-collapse
    alarm on the per-epoch record.

    Bookkeeping runs on CPU: the `[B]`-sized columns are tiny, and gathering
    + `bincount` host-side keeps the meter device-agnostic.
    """

    def __init__(self) -> None:
        self._pred_counts = torch.zeros(8, dtype=torch.long)
        self._target_counts = torch.zeros(8, dtype=torch.long)
        self._n = 0

    def update(
        self,
        top1_idx: torch.Tensor,
        action_target: torch.Tensor,
        non_pass: torch.Tensor,
    ) -> None:
        """Fold one batch in. `top1_idx` is the model's argmax flat action
        index `[B]`; `action_target` is the demonstrated flat index `[B]`
        (-1 on pass frames); `non_pass` is a `[B]` bool mask. Restriction to
        non-pass frames keeps `% 8` well-defined on the target side."""
        npm = non_pass.cpu()
        if not bool(npm.any()):
            return
        self._pred_counts += torch.bincount(top1_idx.cpu()[npm] % 8, minlength=8)
        self._target_counts += torch.bincount(action_target.cpu()[npm] % 8, minlength=8)
        self._n += int(npm.sum())

    def pred_dist(self) -> dict[str, float | None]:
        return self._dist(self._pred_counts)

    def target_dist(self) -> dict[str, float | None]:
        return self._dist(self._target_counts)

    def _dist(self, counts: torch.Tensor) -> dict[str, float | None]:
        """Bucket counts → named-fraction dict, all-None if no frames seen."""
        if self._n == 0:
            return {name: None for name in _SUB_NAMES}
        return {
            name: float(counts[i].item() / self._n)
            for i, name in enumerate(_SUB_NAMES)
        }
