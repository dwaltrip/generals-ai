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
