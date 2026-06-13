"""Pure per-frame elim-head metrics over a stratified dump's columns.

The calc behind the stratified report's elim tables and any ad-hoc cross-cell
synthesis: flatten the dump's elim columns to alive (player, frame) pairs, then
reduce. `confusion_counts` is the primitive — argmax(probs) vs the true bin —
from which per-bin recall, precision, and prediction frequency all derive
(`per_bin_metrics`). Recall alone misleads after class-weighting (over-
prediction inflates it); precision is the read it can't move, which is why the
confusion structure is the load-bearing one. Everything here is display-free —
callers format and print.
"""

from __future__ import annotations

import numpy as np


def elim_flat(
    d: dict[str, np.ndarray], sel: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten the elim columns to alive (player, frame) pairs within the frame
    selection `sel`. Returns (hard CE, true bin, probs [M, n_bins], channel
    index 0..7) — channel 0 is the perspective (self), 1..7 opponents."""
    mask = d["elim_alive_mask"][sel]                       # [n, 8] bool
    ce = d["elim_ce"][sel][mask]                           # [M]
    bins = d["elim_bin_target"][sel][mask]                 # [M]
    probs = d["elim_probs"][sel][mask].astype(np.float32)  # [M, n_bins]
    ch = np.broadcast_to(np.arange(8), mask.shape)[mask]   # [M]
    return ce, bins, probs, ch


def elim_stats(
    ce: np.ndarray, bins: np.ndarray, probs: np.ndarray
) -> dict[str, float] | None:
    """Per-group elim metrics over flattened alive pairs. `top1` and `pred_h`
    are tax-free (argmax / softmax spread); `ce` is the tax-confounded hard CE.
    None when the group is empty."""
    n = int(ce.shape[0])
    if n == 0:
        return None
    return {
        "n": n,
        "ce": float(ce.mean()),
        "top1": float((probs.argmax(axis=1) == bins).mean()),
        "pred_h": float(-(probs * np.log(probs + 1e-12)).sum(axis=1).mean()),
    }


def confusion_counts(
    probs: np.ndarray, bins: np.ndarray, n_bins: int
) -> np.ndarray:
    """The [n_bins, n_bins] confusion-count matrix `C[true, pred]` over
    argmax(probs) vs the true bin, on flattened alive pairs. The primitive
    behind per-bin recall, precision, and prediction frequency — see
    `per_bin_metrics`. Empty input yields an all-zero matrix."""
    pred = probs.argmax(axis=1)
    flat = bins.astype(np.int64) * n_bins + pred.astype(np.int64)
    return np.bincount(flat, minlength=n_bins * n_bins).reshape(n_bins, n_bins)


def per_bin_metrics(C: np.ndarray) -> dict[str, np.ndarray]:
    """Per-bin reductions of a confusion-count matrix `C[true, pred]` (from
    `confusion_counts`), each a length-`n_bins` float array:

    - `recall[b]`    = C[b,b] / support[b]      — true-bin-b pairs called b
    - `precision[b]` = C[b,b] / pred_total[b]   — calls of b that were right
    - `pred_freq[b]` = pred_total[b] / N        — how often b is predicted
    - `support[b]`   = C[b,:].sum()             — true-bin-b base count

    Recall is what class-weighting inflates via over-prediction; precision is
    the orthogonal read it can't directly move. Zero-denominator bins (a bin
    never predicted has undefined precision) come back NaN.
    """
    C = C.astype(np.float64)
    total = C.sum()
    support = C.sum(axis=1)     # true-bin counts
    pred_total = C.sum(axis=0)  # predicted-bin counts
    diag = np.diag(C)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = diag / support
        precision = diag / pred_total
        pred_freq = pred_total / total
    return {
        "recall": recall,
        "precision": precision,
        "pred_freq": pred_freq,
        "support": support,
    }


def bin_labels(meta: dict, n_bins: int) -> list[str]:
    """Human range labels for the bins, from the dump's `elim_bin_edges` meta;
    falls back to plain indices when edges are absent."""
    edges = meta.get("elim_bin_edges")
    if not edges:
        return [str(b) for b in range(n_bins)]
    labels, lo = [], 0
    for e in edges:
        labels.append(f"[{lo},{e})")
        lo = e
    labels.append(f"[>={lo}/never]")
    return labels
