"""Tie-aware accuracy (sliced by surrender status), the none-ceiling reference, the
error-vs-margin curve, and weight inspection.

All accuracies are **unmasked** — the decision is `argmax` over all 8 logits (lowest-
index tiebreak, matching the `label` rule), and picking a dead player counts as
wrong. That's the honest read of whether the encoding taught exclusion; masking
would hide exactly the failure the toy studies (6.18-1 §6, Appendix A).

The headline blends two difficulties (6.18-4 §8 #1), so every accuracy is reported
over three slices:
  - `` (all)            — the whole eval set.
  - `_nonsurr_mixed`    — `n_alive < 8 & ~surr_frame`: the **notch** alone (exclude
                          the `army==0` eliminated). A capacity question.
  - `_surr`             — the surrender subpopulation: the **information ceiling**
                          (`none`/`capture_status` can't exclude `army>0` surrendered
                          players under the `alive` target). Read against `none_ceilings`.

Two accuracy flavors (~12.7% of frames have an exact army tie, 6.18-1 §6):
  - `acc_full`  — `argmax == label` (canonical lowest-index). Folds in tiebreak luck.
  - `acc_inset` — `argmax ∈ argmin_set` (credits identifying *a* true minimum).
`acc_strict` (over `margin > ε`) is reported on the all slice only.
"""

from __future__ import annotations

import numpy as np
import torch

from training.analysis.argmin_toy.train import MetricsFn
from training.analysis.fq.frame_table import FrameTable


# Default error-vs-margin bin edges, in raw-army units (margin is the gap between
# the two lowest in-set armies). Errors should concentrate in the small-margin bins.
DEFAULT_MARGIN_BINS = [0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0, float("inf")]


def _frac(t: torch.Tensor) -> float:
    """Mean of a 1-D bool/float tensor; NaN on an empty slice (so a slice with no
    frames reads as missing, not 0)."""
    return float(t.float().mean().item()) if t.numel() else float("nan")


def make_metrics_fn(eps: float = 0.0) -> MetricsFn:
    """A per-epoch metric fn (closes over the strict-margin threshold `eps`). `eps`
    is on the raw-army `margin` column: `eps=0` excludes only exact ties; `eps≥1`
    excludes the minimal-gap tail."""

    def fn(scores: torch.Tensor, target: torch.Tensor, aux: dict) -> dict[str, float]:
        pred = scores.argmax(dim=1)
        argmin_set = aux["argmin_set"]      # [N, 8] bool
        margin = aux["margin"]              # [N] raw-army gap; -1 when <2 in-set
        n_alive = aux["n_alive"]            # [N]
        surr = aux["surr_frame"].bool()     # [N]
        n = pred.shape[0]

        correct = pred == target
        inset = argmin_set[torch.arange(n), pred]
        slices = {
            "": torch.ones(n, dtype=torch.bool),
            "_surr": surr,
            "_nonsurr_mixed": (n_alive < 8) & ~surr,
        }
        out: dict[str, float] = {}
        for sfx, m in slices.items():
            out[f"acc_full{sfx}"] = _frac(correct[m])
            out[f"acc_inset{sfx}"] = _frac(inset[m])

        strict_sel = margin > eps
        out["acc_strict"] = _frac(correct[strict_sel])
        return out

    return fn


def none_ceilings(val_t: FrameTable, target: str) -> dict[str, float]:
    """Per-slice accuracy ceiling for an army-only (`none`) model on `val_t`.

    The best `none` can do is carve the army-zero notch and then pick the smallest
    `army>0` — `argmin-over-(army>0)`. Its ceiling per slice is that pick's agreement
    with the target `label`. Under `target='army_pos'` the label *is*
    `argmin-over-(army>0)`, so the ceiling is ~100%; under `'alive'` the `_surr`
    slice ceilings well below 100% (the §5 ~57.6% split-specific figure) because the
    surrendered `army>0` player is sometimes the smallest-positive and can't be
    excluded. Drawn as the reference line so a floored cell reads as "at ceiling,"
    not "broken" (6.18-4 §8 #5)."""
    army = val_t.cols["army_sim"].astype(np.int64)
    label = val_t.cols["label"]
    n_alive = val_t.cols["n_alive"]
    surr = val_t.cols["surr_frame"].astype(bool)

    armypos_argmin = np.where(army > 0, army, np.iinfo(np.int64).max).argmin(1)
    agree = armypos_argmin == label
    slices = {
        "all": np.ones(label.shape[0], dtype=bool),
        "surr": surr,
        "nonsurr_mixed": (n_alive < 8) & ~surr,
    }
    return {
        name: float(agree[m].mean()) if m.any() else float("nan")
        for name, m in slices.items()
    }


def tie_stats(aux: dict) -> dict[str, float]:
    """Exact-tie fraction over frames with ≥2 in-set members (`margin >= 0`). A data
    property, constant across epochs — reported once per cell, not per epoch."""
    margin = aux["margin"]
    ge2 = margin >= 0
    tie_frac = (margin[ge2] == 0).float().mean().item() if bool(ge2.any()) else float("nan")
    return {"exact_tie_frac": tie_frac, "n_ge2_alive": int(ge2.sum())}


def error_vs_margin(
    scores: torch.Tensor, target: torch.Tensor, aux: dict, bins: list[float] | None = None
) -> list[dict]:
    """Error rate binned by `margin`. Errors concentrating at small margins is the
    sharpness of the learned comparator (6.18-1 §6). Frames with `margin < 0`
    (<2 in-set) fall out by construction."""
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
