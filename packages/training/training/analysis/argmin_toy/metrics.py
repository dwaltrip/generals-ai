"""Tie-aware accuracy, the error-vs-margin curve, and weight inspection.

All metrics are **unmasked** — the decision is `argmax` over all 8 logits (lowest-
index tiebreak, matching the `label` rule), and picking a dead player counts as
wrong. That's the honest read of whether the encoding taught exclusion; masking
would hide exactly the failure the toy studies (6.18-1 §6, Appendix A).

Three accuracies, because ~12.7% of frames have an exact army tie (6.18-1 §6):
  - `acc_full`   — `argmax == label` (canonical lowest-index). Folds in tiebreak luck.
  - `acc_strict` — `acc_full` over frames with `margin > ε` (raw-army units). The
    honest comparator measure: the gap the model must actually resolve.
  - `acc_inset`  — `argmax ∈ argmin_set` (credits identifying *a* true minimum).
`acc_inset` and `acc_strict` are the load-bearing measures; `acc_full` running
~10% below `acc_inset` is the expected tie gap, not a representational failure.
"""

from __future__ import annotations

import torch

from training.analysis.argmin_toy.train import MetricsFn


# Default error-vs-margin bin edges, in raw-army units (margin is the gap between
# the two lowest alive armies). Errors should concentrate in the small-margin bins.
DEFAULT_MARGIN_BINS = [0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0, float("inf")]


def make_metrics_fn(eps: float = 0.0) -> MetricsFn:
    """A per-epoch metric fn (closes over the strict-margin threshold `eps`). `eps`
    is on the raw-army `margin` column: `eps=0` excludes only exact ties; `eps≥1`
    excludes the minimal-gap tail."""

    def fn(scores: torch.Tensor, target: torch.Tensor, aux: dict) -> dict[str, float]:
        pred = scores.argmax(dim=1)
        argmin_set = aux["argmin_set"]  # [N, 8] bool
        margin = aux["margin"]          # [N] raw-army gap; -1 when <2 alive
        n = pred.shape[0]

        full = (pred == target).float().mean().item()
        in_set = argmin_set[torch.arange(n), pred].float().mean().item()
        strict_sel = margin > eps
        strict = (
            (pred[strict_sel] == target[strict_sel]).float().mean().item()
            if bool(strict_sel.any())
            else float("nan")
        )
        return {"acc_full": full, "acc_strict": strict, "acc_inset": in_set}

    return fn


def tie_stats(aux: dict) -> dict[str, float]:
    """Exact-tie fraction over frames with ≥2 alive (`margin >= 0`). A data property,
    constant across epochs — reported once per cell, not per epoch."""
    margin = aux["margin"]
    ge2 = margin >= 0
    tie_frac = (margin[ge2] == 0).float().mean().item() if bool(ge2.any()) else float("nan")
    return {"exact_tie_frac": tie_frac, "n_ge2_alive": int(ge2.sum())}


def error_vs_margin(
    scores: torch.Tensor, target: torch.Tensor, aux: dict, bins: list[float] | None = None
) -> list[dict]:
    """Error rate binned by `margin`. Errors concentrating at small margins is the
    sharpness of the learned comparator (6.18-1 §6). Frames with `margin < 0`
    (<2 alive) fall out by construction."""
    edges = DEFAULT_MARGIN_BINS if bins is None else bins
    pred = scores.argmax(dim=1)
    margin = aux["margin"]
    correct = pred == target
    out: list[dict] = []
    for lo, hi in zip(edges, edges[1:], strict=False):
        sel = (margin >= lo) & (margin < hi)
        err = float(1.0 - correct[sel].float().mean().item()) if bool(sel.any()) else None
        out.append({"lo": lo, "hi": hi, "n": int(sel.sum()), "error": err})
    return out


def weight_report(inspect: dict) -> dict:
    """JSON-safe summary of a model's `.inspect()` dump, with the reference-recovery
    scalars: A1's `neg_identity_residual` (`‖W+I‖/‖I‖`, ~0 when `W ≈ −I`) at `C=1`
    and the symmetric-subspace residual; D1's `(A, B)`; E1's `w`. B1 → `{}`."""
    out: dict = {}
    if "W" in inspect:  # A1
        W = inspect["W"]
        out["sym_subspace_residual"] = inspect["sym_subspace_residual"]
        if W.shape[0] == W.shape[1]:  # square => C=1, the only case with a clean -I
            eye = torch.eye(W.shape[0], dtype=W.dtype)
            out["neg_identity_residual"] = float((W + eye).norm() / eye.norm())
        out["W"] = W.tolist()
    if "A" in inspect:  # D1
        out["A"] = inspect["A"].tolist()
        out["B"] = inspect["B"].tolist()
    if "w" in inspect:  # E1
        out["w"] = None if inspect["w"] is None else inspect["w"].tolist()
    return out
