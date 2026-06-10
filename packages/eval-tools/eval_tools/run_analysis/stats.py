"""Numpy-only statistics helpers for eval-run analysis.

Resampling unit is always the *game*: seats within a game are correlated
(one winner, shared economy), so bootstrap draws game ids, never rows.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import math

import numpy as np


PCTLS = (5, 10, 25, 50, 75, 90, 95)


def summarize(values: Sequence[float]) -> dict:
    """Distribution summary of one metric (nan-aware)."""
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    out = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    for p in PCTLS:
        out[f"p{p}"] = float(np.percentile(arr, p))
    return out


def bootstrap_se_by_game(
    pairs: Sequence[tuple[str, float]], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """(point estimate, bootstrap SE) for the mean-over-games of per-game means.

    `pairs` are (game_id, value); games get equal weight regardless of how
    many rows they contribute.
    """
    by_game: dict[str, list[float]] = defaultdict(list)
    for gid, v in pairs:
        if not math.isnan(v):
            by_game[gid].append(v)
    game_means = np.asarray([float(np.mean(vs)) for vs in by_game.values()])
    if game_means.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, game_means.size, size=(n_boot, game_means.size))
    boot = game_means[idx].mean(axis=1)
    return float(game_means.mean()), float(boot.std(ddof=1))


def sign_test(n_a: int, n_b: int) -> float:
    """Two-sided exact binomial p for n_a vs n_b discordant outcomes."""
    n = n_a + n_b
    if n == 0:
        return 1.0
    k = min(n_a, n_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_paired_diff(
    deltas: Sequence[float], n_boot: int = 2000, seed: int = 0
) -> dict:
    """Mean and bootstrap SE / 95% CI of per-pair deltas (e.g. per-map Δ)."""
    arr = np.asarray([d for d in deltas if not math.isnan(d)], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "se": float(boot.std(ddof=1)),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def median_iqr(values: Sequence[float]) -> str:
    """Render "median [p25–p75]" for a table cell; nan-aware."""
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return "—"
    p25, p50, p75 = (float(np.percentile(arr, p)) for p in (25, 50, 75))
    return f"{p50:.3g} [{p25:.3g}–{p75:.3g}]"
