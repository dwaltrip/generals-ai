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
    metric fn (margin / argmin_set for the tie-aware measures; n_alive / surr_frame
    for the surrender-status slices)."""

    inputs: tuple[str, ...] = ("army", "land", "alive", "captured")
    target: str = "label"
    aux: tuple[str, ...] = ("alive", "margin", "argmin_set", "n_alive", "surr_frame")


@dataclass(frozen=True)
class HParams:
    """Loop hyperparameters. A starting point, not a finding — the notch encodings
    may need more epochs to resolve represent-vs-optimize (6.18-1 Risk #3)."""

    lr: float = 1e-2
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 200
    # Early stop: halt once val_loss hasn't improved by > `min_delta` for `patience`
    # epochs (but never before `min_epochs`). The converged cells (E1_mlp ~ep 7,
    # equivariant ~ep 1) and the stuck cells (linear on the notch, flat at the floor)
    # both plateau, so this trims wasted epochs from both. `patience <= 0` disables
    # it (runs the full `epochs`). We keep — not restore — the final weights and
    # report best + last from the curves, so a late val-loss divergence stays visible.
    patience: int = 30
    min_delta: float = 1e-3
    min_epochs: int = 20
    # Console print cadence only — curves are recorded every epoch regardless (the
    # artifact and early-stop read them); this just thins the per-epoch log. `init`,
    # every `log_every`-th epoch, the final epoch, and the early-stop epoch print.
    log_every: int = 10


@dataclass(frozen=True)
class TrainResult:
    """One run's output: the full per-epoch `curves` (entry 0 = pre-train baseline),
    plus pointers — `best_epoch` is the curve index with the lowest val_loss, and
    `stopped_epoch` is the last epoch actually run (`== epochs` unless early-stopped).
    The model is left at its `stopped_epoch` (last) state; `best_epoch` lets the
    driver report best + last without restoring weights."""

    curves: dict[str, list[float]]
    best_epoch: int
    stopped_epoch: int


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
) -> TrainResult:
    """Train `model` on `train_view`, eval on `val_view` each epoch. Returns the
    per-epoch curves (entry 0 = pre-training baseline) plus the best/last epoch
    pointers, early-stopping once val_loss plateaus (HParams.patience)."""
    optim = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    curves: dict[str, list[float]] = defaultdict(list)

    def record(label: str) -> str:
        """Append this step's losses/metrics to the curves; return the formatted log
        line (the caller decides whether to print it)."""
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
        return f"  {label:>8}  train {tr_loss:.4f}  val {va_loss:.4f}  {metric_str}"

    n_train = train_view[roles.target].shape[0]
    target = train_view[roles.target]
    print(f"  {'epoch':>8}  {'train':>10}  {'val':>10}  metrics")
    print(record("init"))
    best_loss, stale, stopped = curves["val_loss"][0], 0, 0
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
        line = record(f"ep {epoch}")
        stopped = epoch
        shown = epoch % hp.log_every == 0 or epoch == hp.epochs
        if shown:
            print(line)

        if curves["val_loss"][-1] < best_loss - hp.min_delta:
            best_loss, stale = curves["val_loss"][-1], 0
        else:
            stale += 1
        if hp.patience > 0 and epoch >= hp.min_epochs and stale >= hp.patience:
            if not shown:
                print(line)
            print(f"  early stop at ep {epoch} (val_loss flat for {stale})")
            break

    val = curves["val_loss"]
    best_epoch = min(range(len(val)), key=val.__getitem__)  # lowest-val_loss curve index
    return TrainResult(curves=dict(curves), best_epoch=best_epoch, stopped_epoch=stopped)
