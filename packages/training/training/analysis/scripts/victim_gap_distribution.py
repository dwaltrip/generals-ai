#!/usr/bin/env -S uv run python
"""Soft-target calibration for the next_death head: death-time gap distribution
and soft-target peakiness preview.

The 6.20-2 direction names a soft next_death target as the leading experiment but
leaves the *construction* open. This script measures the two things that decide it,
model-free over the val split:

  1. The distribution of consecutive death-tick GAPS among the currently-alive
     players (soonest→2nd, 2nd→3rd). Small gaps = the near-tie frames a soft target
     reclaims; large gaps = frames where the next victim is already determined and a
     soft target should stay ~one-hot. This says how much softening is even on the
     table, and on what fraction of frames.

  2. A PREVIEW of two candidate soft targets across a temperature grid:
       - time-based:  p_i ∝ exp(−(death_i − death_min) / τ)   (τ in ticks)
       - rank-based:  p_i ∝ exp(−rank_i / τ)                  (τ dimensionless)
     reported as mean mass on the hard label (the soonest victim) and mean
     perplexity (effective # players the target spreads over). Split by whether the
     soonest→2nd gap is small vs large — the conditional that separates time-based
     (softens only the genuine near-ties) from rank-based (softens uniformly).

Reuses the Family-A fq plumbing: one FrameTable over the val split carrying the
per-channel `death_tick` + the `alive`/`victim`/`dt` truth, no model needed.

NOTE: this measures the *death* event (surrender-or-capture) over the alive
domain — the calibration that picked τ=15. The shipped next_death target keys on
*board-removal* over the present domain instead; the two differ only on the
surrender window (~1% of frames), so the gap picture — and τ — carry over. Left
death-based as the as-run calibration record.

Usage:
    victim_gap_distribution.py [--manifest PATH] [--max-games N] [--split val|train|all]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from training.analysis.fq.frame_table import (
    GROUND_TRUTH_OBS_CFG,
    FrameTable,
    FrameTableSpec,
    build_frame_table,
    select,
)
from training.analysis.scripts.family_a_victim_baselines import (
    DEFAULT_MANIFEST,
    NO_DEATH,
    _death_tick_deriver,
)
from training.bc.aux_heads.elim_head_meta import ElimHeadVariant
from training.bc.config.targets_config import TargetsConfig
from training.bc.emit_spec import PartialEmitSpec
from training.bc.splits import load_manifest, samples_for_split


# Candidate temperatures. Time grid spans ticks well below to above the median
# time-to-death (~54); rank grid spans peaked→flat for a dimensionless decay.
TAU_TIME = (5.0, 10.0, 25.0, 50.0, 100.0)
TAU_RANK = (0.3, 0.5, 0.7, 1.0, 1.5)
# Gap thresholds (ticks) for the near-tie tail of the CDF.
GAP_THRESHOLDS = (0, 1, 2, 5, 10, 25, 50)
# "Near-tie" cut for the conditional soft-target preview.
SMALL_GAP = 10


def build_spec() -> FrameTableSpec:
    """Minimal table: per-channel death tick + the next_death truth (alive/victim/dt).

    No score derivers — this measurement needs only death timing and aliveness, so
    it skips the whole rate-construction menu the baseline script builds."""
    return FrameTableSpec(
        name="victim_gaps",
        emit=PartialEmitSpec(
            targets=TargetsConfig(
                elim_variant=ElimHeadVariant.NEXT_DEATH, elim_bin_edges=None
            ),
            emit_alive_mask=True,
            attach_sim_frame=True,
        ),
        emit_cols={"alive_mask": "alive", "next_elim_target": "victim", "next_elim_dt": "dt"},
        derivers=[_death_tick_deriver()],
        truth_map={},
    )


def sorted_alive_deaths(t: FrameTable) -> np.ndarray:
    """`[N, 8]` per-frame death ticks of the alive+real channels, sorted ascending,
    non-real slots filled with +inf (sort to the end). Column 0 = soonest death."""
    alive = t.cols["alive"].astype(bool)
    death = t.cols["death_tick"].astype(np.float64)
    real = alive & (death < NO_DEATH)
    d = np.where(real, death, np.inf)
    return np.sort(d, axis=1)


def report_horizon(t: FrameTable) -> None:
    """Time-to-next-death — the absolute horizon (context for the gaps below)."""
    dt = t.cols["dt"]
    print(f"\n# Frames: {dt.size}  games: {t.n_games}")
    cdf = {f"<{q}": round(float((dt < q).mean() * 100), 1) for q in (10, 25, 50, 100, 200)}
    print(f"# time-to-next-death (dt): median={np.median(dt):.0f}  CDF%={cdf}")


def _print_gap_summary(label: str, gap: np.ndarray) -> None:
    """Percentiles + near-tie CDF for one finite gap sample."""
    if gap.size == 0:
        print(f"# {label}: (no frames)")
        return
    pcts = {f"p{p}": int(np.percentile(gap, p)) for p in (10, 25, 50, 75, 90, 95)}
    within = {f"<={th}": round(float((gap <= th).mean() * 100), 1) for th in GAP_THRESHOLDS}
    print(f"# {label}  [{gap.size} frames]  mean={gap.mean():.0f}")
    print(f"    percentiles: {pcts}")
    print(f"    within: {within}")


def _consec_gap(sd: np.ndarray, k: int) -> np.ndarray:
    """Finite values of the rank-k→(k+1) death gap (both ranks must exist)."""
    with np.errstate(invalid="ignore"):  # inf - inf (one rank missing) -> nan, filtered next
        gap = sd[:, k + 1] - sd[:, k]
    return gap[np.isfinite(gap)]


def report_gaps(sd: np.ndarray) -> None:
    """The FULL consecutive death-gap distribution: every rank-k→(k+1) gap, plus all
    pooled. Time-based softening is driven by the soonest gaps, but the deeper ranks
    show how spread the rest of the death order is."""
    n_real = np.isfinite(sd).sum(axis=1)
    print(f"\n# Alive real-death count per frame: "
          f"mean={n_real.mean():.2f}  "
          f">=2: {(n_real >= 2).mean() * 100:.1f}%  >=3: {(n_real >= 3).mean() * 100:.1f}%")

    print("\n## Full consecutive death-gap distribution (ticks)")
    pooled = []
    for k in range(7):  # ranks 0..7 -> up to 7 consecutive gaps
        gap = _consec_gap(sd, k)
        if gap.size == 0:
            continue
        _print_gap_summary(f"rank {k}->{k + 1}", gap)
        pooled.append(gap)
    if pooled:
        _print_gap_summary("ALL consecutive gaps pooled", np.concatenate(pooled))


def report_gaps_by_nalive(t: FrameTable, sd: np.ndarray) -> None:
    """The soonest→2nd gap (the gap the soft target leans on most) bucketed by how
    many players are currently alive — the softmax support size. Exactly one alive
    player (the winner) has no real future death, so n_alive == n_real + 1; the gap
    needs n_real >= 2, i.e. n_alive >= 3."""
    n_alive = t.cols["alive"].astype(bool).sum(axis=1)
    gap0 = sd[:, 1] - sd[:, 0]
    print("\n## soonest->2nd gap bucketed by # players currently alive")
    for n in range(2, 9):
        sel = n_alive == n
        if not sel.any():
            continue
        with np.errstate(invalid="ignore"):
            g = gap0[sel]
        g = g[np.isfinite(g)]
        frac = sel.mean() * 100
        _print_gap_summary(f"n_alive={n}  ({frac:.1f}% of frames)", g)


def _soft_stats(logits: np.ndarray, real: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given per-channel `logits` (`-inf` off-support) and a `real` mask, return
    `(mass_on_soonest, perplexity)` per frame for the softmax over the real set.

    `mass_on_soonest` = the probability the target puts on the hard label (the
    soonest-death channel). `perplexity = exp(entropy)` = effective # players the
    soft target spreads over (1.0 = one-hot, n_real = uniform)."""
    z = np.where(real, logits, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    w = np.exp(z)
    p = w / w.sum(axis=1, keepdims=True)
    soonest_mass = p.max(axis=1)  # soonest death has the largest weight under both schemes
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
    return soonest_mass, np.exp(ent)


def report_soft_preview(t: FrameTable, sd: np.ndarray) -> None:
    """Mean mass-on-hard-label and perplexity for each candidate (scheme, τ),
    overall and split by near-tie vs clear-gap frames."""
    alive = t.cols["alive"].astype(bool)
    death = t.cols["death_tick"].astype(np.float64)
    real = alive & (death < NO_DEATH)
    d = np.where(real, death, 0.0)
    dmin = np.where(real, death, np.inf).min(axis=1, keepdims=True)

    # Rank among real deaths (0 = soonest); non-real get a large rank, masked anyway.
    big = np.inf
    order = np.argsort(np.where(real, death, big), axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(8)[None, :], axis=1)

    # Conditional split on the soonest->2nd gap (only frames with >=2 real deaths).
    gap1 = sd[:, 1] - sd[:, 0]
    has2 = np.isfinite(gap1)
    near = has2 & (gap1 <= SMALL_GAP)
    clear = has2 & (gap1 > SMALL_GAP)

    def line(name: str, soonest: np.ndarray, ppl: np.ndarray) -> None:
        def fmt(sel: np.ndarray) -> str:
            if not sel.any():
                return "    --"
            return f"{soonest[sel].mean() * 100:5.1f}% / {ppl[sel].mean():4.2f}"
        print(f"  {name:>14}:  all {fmt(np.ones_like(near))}   "
              f"near-tie {fmt(near)}   clear {fmt(clear)}")

    print("\n# Soft-target preview — mass-on-hard-label% / perplexity")
    print(f"#   near-tie = soonest->2nd gap <= {SMALL_GAP} ticks "
          f"({near.mean() * 100:.1f}% of frames); "
          f"clear = gap > {SMALL_GAP} ({clear.mean() * 100:.1f}%)")
    print("#   (one-hot = 100% / 1.00; flatter target = lower% / higher perplexity)")

    print("\n  time-based  p_i ∝ exp(−(deathᵢ − death_min)/τ)")
    for tau in TAU_TIME:
        logits = -(d - dmin) / tau
        line(f"τ={tau:g}", *_soft_stats(logits, real))

    print("\n  rank-based  p_i ∝ exp(−rankᵢ/τ)")
    for tau in TAU_RANK:
        logits = -rank.astype(np.float64) / tau
        line(f"τ={tau:g}", *_soft_stats(logits, real))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="split manifest JSON")
    ap.add_argument(
        "--max-games", type=int, default=None, help="cap to first N games (fast iteration)"
    )
    ap.add_argument(
        "--split", choices=("val", "train", "all"), default="val",
        help="which manifest split to measure. 'all' = train+val (this is pure "
             "model-free measurement, so train leakage is irrelevant; more games = tighter tails).",
    )
    args = ap.parse_args()

    spec = build_spec()
    man = load_manifest(args.manifest)
    root = Path(man["intermediate_root"])
    splits = ("train", "val") if args.split == "all" else (args.split,)
    samples = [s for sp in splits for s in samples_for_split(man, sp, root)]
    print(f"building victim-gap table: {len(samples)} {args.split} perspectives "
          f"({'all games' if args.max_games is None else f'first {args.max_games} games'})")
    t = build_frame_table(spec, samples, GROUND_TRUTH_OBS_CFG, args.max_games)

    # Drop winner-tail frames (no real next death); these carry victim == -1.
    t = select(t, t.cols["victim"] >= 0)

    sd = sorted_alive_deaths(t)
    report_horizon(t)
    report_gaps(sd)
    report_gaps_by_nalive(t, sd)
    report_soft_preview(t, sd)


if __name__ == "__main__":
    main()
