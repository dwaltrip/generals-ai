"""Analyze week-over-week stars deltas to look for a rank-decay signature.

Hypothesis: inactive players form a tight mode in the negative tail of the
week-over-week stars-delta distribution (their drop reflects pure decay).
Active players form a wider, higher-centered distribution. If we can see the
inactive mode clearly, we can estimate the decay rate and use it to infer
per-player, per-week activity.

Usage:
    uv run --with matplotlib scripts/analyze_stars_decay.py [season]

Default season: 42. Reads data/leaderboards/season-{N}.json, writes plot to
scripts/out/stars_decay_season-{N}.png.
"""

import argparse
from collections import defaultdict
from itertools import pairwise
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports, reportMissingModuleSource]

# Provided via `uv run --with matplotlib` per the module docstring.
from utils.docstring import doc_summary


METRICS = ["ffa", "ffawin", "ffacombat", "ffakills"]
DATA_DIR = Path("data/leaderboards")
OUT_DIR = Path("scripts/out")

# Window assumed to bracket the inactive-player mode based on first-pass plot.
INACTIVE_WINDOW = (-8.0, -3.0)

# Tighter range used to gate the delta1-suggests-inactive filter for the
# second-week-followup analysis. Should sit inside INACTIVE_WINDOW and
# exclude the active-but-losing tail and the active-near-zero region.
INACTIVE_D1_RANGE = (-5.0, -2.0)


def load_trajectories(path: Path, metric: str) -> dict[str, dict[int, float]]:
    """Return {username: {week: stars}} for the given metric."""
    with path.open() as f:
        data = json.load(f)
    rankings = data["rankings"]
    traj: dict[str, dict[int, float]] = defaultdict(dict)
    for week in range(1, len(rankings)):
        for entry in rankings[week][metric]:
            traj[entry["username"]][week] = entry["stars"]
    return traj


def consecutive_deltas(
    traj: dict[str, dict[int, float]],
) -> list[tuple[float, float]]:
    """For each player, yield (stars_at_week_N, delta_to_week_N+1) pairs
    where the player appears in both consecutive weeks."""
    pairs: list[tuple[float, float]] = []
    for weeks in traj.values():
        sorted_weeks = sorted(weeks.keys())
        for w1, w2 in pairwise(sorted_weeks):
            if w2 == w1 + 1:
                s1 = weeks[w1]
                s2 = weeks[w2]
                pairs.append((s1, s2 - s1))
    return pairs


def consecutive_triplets(
    traj: dict[str, dict[int, float]],
) -> list[tuple[float, float]]:
    """Return (delta1, delta2) for each player who appears in three
    consecutive weeks. delta1 is change during the middle week; delta2
    is change during the third week.

    If delta1 is in the inactive band, the player was likely inactive in
    the middle week, meaning the rating at the start of the third week
    is already post-decay (no active-game residual). delta2 is therefore
    a cleaner measurement of pure decay than delta1 would be.
    """
    out: list[tuple[float, float]] = []
    for weeks in traj.values():
        sw = sorted(weeks.keys())
        for w_a, w_b, w_c in zip(sw, sw[1:], sw[2:], strict=False):
            if w_b == w_a + 1 and w_c == w_b + 1:
                d1 = weeks[w_b] - weeks[w_a]
                d2 = weeks[w_c] - weeks[w_b]
                out.append((d1, d2))
    return out


def estimate_decay(deltas: list[float]) -> dict[str, float]:
    """Estimate decay rate from the inactive-mode window."""
    lo, hi = INACTIVE_WINDOW
    in_window = sorted(d for d in deltas if lo <= d <= hi)
    n = len(in_window)
    if not n:
        return {"n": 0}
    median = in_window[n // 2]
    mean = sum(in_window) / n
    floor = min(deltas)
    # Tight band around the floor: how concentrated is the left edge?
    near_floor = sum(1 for d in deltas if d <= floor + 0.5)
    return {
        "n": float(n),
        "median": median,
        "mean": mean,
        "floor": floor,
        "near_floor_count": float(near_floor),
        "near_floor_pct": near_floor / len(deltas),
    }


def summarize(metric: str, pairs: list[tuple[float, float]]) -> dict[str, float]:
    deltas = [d for _, d in pairs]
    n = len(deltas)
    if not n:
        print(f"  {metric}: no consecutive-week pairs", file=sys.stderr)
        return {}
    deltas_sorted = sorted(deltas)
    pos = sum(1 for d in deltas if d > 0)
    zero = sum(1 for d in deltas if d == 0)
    neg = n - pos - zero
    print(f"  {metric}: n={n} pairs", file=sys.stderr)
    print(
        f"    pos={pos} ({pos / n:.0%}) "
        f"zero={zero} ({zero / n:.0%}) "
        f"neg={neg} ({neg / n:.0%})",
        file=sys.stderr,
    )
    print(
        f"    delta percentiles: "
        f"p5={deltas_sorted[int(0.05 * n)]:.2f} "
        f"p25={deltas_sorted[int(0.25 * n)]:.2f} "
        f"p50={deltas_sorted[n // 2]:.2f} "
        f"p75={deltas_sorted[int(0.75 * n)]:.2f} "
        f"p95={deltas_sorted[int(0.95 * n)]:.2f}",
        file=sys.stderr,
    )
    decay = estimate_decay(deltas)
    if decay.get("n"):
        print(
            f"    inactive-window {INACTIVE_WINDOW}: "
            f"n={int(decay['n'])} "
            f"median={decay['median']:.2f} "
            f"mean={decay['mean']:.2f}",
            file=sys.stderr,
        )
        print(
            f"    floor={decay['floor']:.2f} "
            f"within 0.5 of floor: {int(decay['near_floor_count'])} "
            f"({decay['near_floor_pct']:.1%} of all deltas)",
            file=sys.stderr,
        )
    return decay


def plot(
    season: int,
    by_metric: dict[str, list[tuple[float, float]]],
    decay_by_metric: dict[str, dict[str, float]],
) -> Path:
    fig, axes = plt.subplots(len(METRICS), 2, figsize=(13, 3.8 * len(METRICS)))
    if len(METRICS) == 1:
        axes = [axes]

    for row, metric in enumerate(METRICS):
        pairs = by_metric[metric]
        stars_vals = [s for s, _ in pairs]
        deltas = [d for _, d in pairs]
        decay = decay_by_metric.get(metric, {})

        ax_hist = axes[row][0]
        ax_hist.hist(deltas, bins=80, color="steelblue", edgecolor="white")
        ax_hist.axvline(0, color="black", linestyle="--", linewidth=0.8)
        if decay.get("n"):
            ax_hist.axvline(
                decay["median"],
                color="crimson",
                linestyle="-",
                linewidth=1.2,
                label=f"inactive-mode median: {decay['median']:.2f}",
            )
            ax_hist.axvline(
                decay["floor"],
                color="darkorange",
                linestyle=":",
                linewidth=1.2,
                label=f"floor: {decay['floor']:.2f}",
            )
            ax_hist.legend(loc="upper right", fontsize=8)
        ax_hist.set_title(f"{metric}: week-over-week stars delta")
        ax_hist.set_xlabel("delta (stars)")
        ax_hist.set_ylabel("count")

        ax_scatter = axes[row][1]
        ax_scatter.scatter(stars_vals, deltas, s=4, alpha=0.25, color="steelblue")
        ax_scatter.axhline(0, color="black", linestyle="--", linewidth=0.8)
        if decay.get("n"):
            ax_scatter.axhline(
                decay["median"], color="crimson", linestyle="-", linewidth=1.0
            )
            ax_scatter.axhline(
                decay["floor"], color="darkorange", linestyle=":", linewidth=1.0
            )
        ax_scatter.set_title(f"{metric}: delta vs current stars")
        ax_scatter.set_xlabel("stars at week N")
        ax_scatter.set_ylabel("delta to week N+1")

    fig.suptitle(f"Season {season} stars-decay analysis", fontsize=14)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"stars_decay_season-{season}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_followup(season: int, by_metric: dict[str, list[float]]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()
    for i, metric in enumerate(METRICS):
        ax = axes_flat[i]
        d2 = by_metric.get(metric, [])
        if not d2:
            ax.set_title(f"{metric}: no triplets")
            continue
        ds = sorted(d2)
        n = len(ds)
        median = ds[n // 2]
        mean = sum(ds) / n
        ax.hist(d2, bins=40, color="crimson", edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(
            median,
            color="navy",
            linewidth=1.4,
            label=f"median: {median:.2f}",
        )
        ax.axvline(
            mean,
            color="darkorange",
            linestyle=":",
            linewidth=1.4,
            label=f"mean: {mean:.2f}",
        )
        ax.set_title(f"{metric}: d2 | d1 ∈ {INACTIVE_D1_RANGE}  (n={n})")
        ax.set_xlabel("delta during 2nd consecutive inactive week (stars)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"Season {season}: pure-decay signal from 2nd consecutive inactive week",
        fontsize=14,
    )
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"stars_decay_followup_season-{season}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=doc_summary(__doc__))
    parser.add_argument("season", type=int, nargs="?", default=42)
    args = parser.parse_args()

    path = DATA_DIR / f"season-{args.season}.json"
    if not path.exists():
        parser.error(f"missing {path}")

    print(f"loading {path}", file=sys.stderr)
    by_metric: dict[str, list[tuple[float, float]]] = {}
    decay_by_metric: dict[str, dict[str, float]] = {}
    followup_by_metric: dict[str, list[float]] = {}
    for metric in METRICS:
        traj = load_trajectories(path, metric)
        pairs = consecutive_deltas(traj)
        by_metric[metric] = pairs
        print(f"{metric}: {len(traj)} unique players", file=sys.stderr)
        decay_by_metric[metric] = summarize(metric, pairs)

        triplets = consecutive_triplets(traj)
        lo, hi = INACTIVE_D1_RANGE
        d2_filtered = [d2 for d1, d2 in triplets if lo <= d1 <= hi]
        followup_by_metric[metric] = d2_filtered
        if d2_filtered:
            ds = sorted(d2_filtered)
            n = len(ds)
            mean = sum(ds) / n
            std = (sum((x - mean) ** 2 for x in ds) / n) ** 0.5
            print(
                f"    triplets: {len(triplets)}  "
                f"with d1 in {INACTIVE_D1_RANGE}: {n}  "
                f"d2 median={ds[n // 2]:.2f}  mean={mean:.2f}  std={std:.2f}",
                file=sys.stderr,
            )

    out = plot(args.season, by_metric, decay_by_metric)
    print(f"wrote {out}", file=sys.stderr)
    out2 = plot_followup(args.season, followup_by_metric)
    print(f"wrote {out2}", file=sys.stderr)


if __name__ == "__main__":
    main()
