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
        # Multiplicative mask rather than a bool-mask gather (`ent[non_pass]`)
        # — the gather returns garbage indices on MPS (TODO(mps-val-crash)).
        self._sum += float((ent * non_pass).sum().item())
        self._n += int(non_pass.sum())

    def mean(self) -> float | None:
        """Mean entropy in nats; `None` if no non-pass frames were seen."""
        return self._sum / self._n if self._n > 0 else None
