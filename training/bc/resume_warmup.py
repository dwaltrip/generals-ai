"""Linear LR warmup for legacy-checkpoint resumes.

A deliberately narrow module — the workaround for cold-restarting AdamW from a
legacy bare-`state_dict` checkpoint, not a general LR-schedule framework.

The problem it addresses: a legacy checkpoint saved only the model weights, not
the optimizer. AdamW scales each parameter's step by `1/sqrt(v_t)`, where `v_t`
is an exponential moving average of squared gradients (the per-parameter
variance estimate, β₂=0.999 → ~700-batch half-life). On resume from a legacy
checkpoint that `v_t` restarts at zero, so the first steps are badly calibrated
— effectively far too large. `v_t` can only re-accumulate by taking real
optimizer steps, so the fix isn't to wait, it's to take those steps gently:
ramp the learning rate from near-zero up to the target over the first `N`
batches of the resumed segment, capping the damage the cold optimizer can do
while its variance estimate fills back in.
"""

from __future__ import annotations

import torch


class WarmupSchedule:
    """Linear LR ramp over the first `n_batches` of a resumed segment.

    `step(optim)` is called once at the top of each batch. The learning rate
    set on every param group is `target_lr · (t+1)/N` for batch `t` (0-indexed),
    capped at `target_lr` — so it climbs from `target_lr/N` on the first batch to
    `target_lr` on the Nth, then holds for the rest of the run.

    The batch counter `t` is transient (it lives here, not on `TrainingState`)
    and spans the whole resumed *segment*, not a single epoch: one instance is
    built per resume and threaded through every epoch, so `t` keeps climbing
    across epoch boundaries. `N` counts from resume, not from the checkpoint's
    prior training history — only fresh optimizer steps re-warm `v_t`.
    """

    def __init__(self, target_lr: float, n_batches: int):
        self.target_lr = target_lr
        self.n_batches = n_batches
        self.t = 0  # batches stepped so far in this segment

    def step(self, optim: torch.optim.Optimizer) -> None:
        """Set the warmed-up LR for the upcoming batch, then advance the counter."""
        frac = min(1.0, (self.t + 1) / self.n_batches)
        lr = self.target_lr * frac
        for group in optim.param_groups:
            group["lr"] = lr
        self.t += 1
