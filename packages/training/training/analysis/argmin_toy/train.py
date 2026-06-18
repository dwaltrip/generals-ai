"""The table-reading training loop — the seed of the 6.17-3 engine.

Deliberately minimal: a table view (a dict of column-name → `[N, ...]` tensor) plus
an explicit `Roles` assignment (which columns are inputs / target / metric-aux),
and a loop. Not `ProbeCache`, not a device-bundle abstraction — promote it only
when the real-aux-head consumer pulls it (6.18-1 §5, Risk #5).

**Loss is plain `F.cross_entropy` over all 8 logits — unmasked.** This is the spine
of the experiment, not a configurable knob: handing the model the `alive` mask as
`−inf` would do the dead-player exclusion *for* it and collapse the encoding axis
(6.18-1 Appendix A). Every frame has ≥1 alive player, so every label is valid — no
`ignore_index`. Per-epoch curves mirror `train_probe_head` (entry 0 = pre-train
baseline; `train_loss` / `val_loss` / `val_<metric>` keys) so output stays
comparable with probe runs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# A metric fn: (logits[N,8], target[N], aux{name: [N,...]}) -> {name: value}.
MetricsFn = Callable[[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]], dict[str, float]]
View = dict[str, torch.Tensor]


@dataclass(frozen=True)
class Roles:
    """How the loop reads a table view: `inputs` are passed positionally to
    `model(*inputs)`, `target` feeds the CE, `aux` columns are handed to the
    metric fn (margin / argmin_set / alive for the tie-aware measures)."""

    inputs: tuple[str, ...] = ("army", "land", "alive")
    target: str = "label"
    aux: tuple[str, ...] = ("alive", "margin", "argmin_set")


@dataclass(frozen=True)
class HParams:
    """Loop hyperparameters. A starting point, not a finding — `zero_overload` may
    need more epochs to resolve represent-vs-optimize (6.18-1 Risk #3)."""

    lr: float = 1e-2
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 200


@torch.no_grad()
def _forward_view(model: nn.Module, view: View, roles: Roles, batch_size: int,
                  device: torch.device) -> torch.Tensor:
    """Run `model` over a full view in minibatches; return `[N, 8]` logits on CPU."""
    model.eval()
    n = view[roles.target].shape[0]
    out: list[torch.Tensor] = []
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        inputs = [view[name][sl].to(device) for name in roles.inputs]
        out.append(model(*inputs).cpu())
    return torch.cat(out, dim=0)


def train_argmin_model(
    model: nn.Module,
    train_view: View,
    val_view: View,
    roles: Roles,
    metrics_fn: MetricsFn,
    hp: HParams,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, list[float]]:
    """Train `model` on `train_view`, eval on `val_view` each epoch. Returns
    per-epoch curves (entry 0 = pre-training baseline)."""
    optim = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    curves: dict[str, list[float]] = defaultdict(list)

    def record(label: str) -> None:
        tr_scores = _forward_view(model, train_view, roles, hp.batch_size, device)
        va_scores = _forward_view(model, val_view, roles, hp.batch_size, device)
        tr_loss = F.cross_entropy(tr_scores, train_view[roles.target]).item()
        va_loss = F.cross_entropy(va_scores, val_view[roles.target]).item()
        aux = {k: val_view[k] for k in roles.aux}
        metrics = metrics_fn(va_scores, val_view[roles.target], aux)
        curves["train_loss"].append(tr_loss)
        curves["val_loss"].append(va_loss)
        for k, v in metrics.items():
            curves[f"val_{k}"].append(v)
        metric_str = "  ".join(f"{k} {v:.3f}" for k, v in metrics.items())
        print(f"  {label:>8}  train {tr_loss:.4f}  val {va_loss:.4f}  {metric_str}")

    n_train = train_view[roles.target].shape[0]
    target = train_view[roles.target]
    print(f"  {'epoch':>8}  {'train':>10}  {'val':>10}  metrics")
    record("init")
    for epoch in range(1, hp.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=rng)
        for start in range(0, n_train, hp.batch_size):
            idx = perm[start : start + hp.batch_size]
            inputs = [train_view[name][idx].to(device) for name in roles.inputs]
            optim.zero_grad()
            loss = F.cross_entropy(model(*inputs), target[idx].to(device))
            loss.backward()
            optim.step()
        record(f"ep {epoch}")

    return dict(curves)
