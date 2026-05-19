"""Pass-rate sweep — sample N parsed games, count `-1` sentinels in
`actions_source` to settle the initial `μ` pass-head loss weight.

Plan napkin: 33-45% pass frames; one observed sample was 56.7%. This
script reports the corpus aggregate + by-month breakdown.

Run from repo root:
    uv run python training/investigations/5_18_pass_rate_sweep.py
    uv run python training/investigations/5_18_pass_rate_sweep.py --n-samples 50000
"""

import argparse
from collections import defaultdict
import time

import numpy as np

from _helpers import (
    load_replay_months,
    meta_path_for,
    replay_id_from_path,
    sample_files,
)

from utils.distribution import print_distribution


DEFAULT_N = 10_000
DEFAULT_SEED = 518


def compute_pass_stats(sim_path) -> tuple[int, int]:
    """Returns (pass_count, total_entries) over perspective slots only."""
    with np.load(meta_path_for(sim_path)) as meta:
        pids = meta["perspective_player_ids"]
    with np.load(sim_path) as data:
        actions = data["actions_source"][pids]
    return int((actions == -1).sum()), int(actions.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print(f"Sampling {args.n_samples} sim files (seed={args.seed}) ...")
    files = sample_files("sim", args.n_samples, args.seed)
    print(f"  got {len(files)} files")

    print("Loading replay month index from sqlite ...")
    t0 = time.perf_counter()
    months = load_replay_months()
    print(f"  loaded {len(months)} months in {time.perf_counter() - t0:.2f}s")

    print("Computing pass rates (perspective slots only) ...")
    t0 = time.perf_counter()
    rates: list[float] = []
    by_month_passes: dict[str, int] = defaultdict(int)
    by_month_entries: dict[str, int] = defaultdict(int)
    by_month_games: dict[str, int] = defaultdict(int)
    total_passes = 0
    total_entries = 0
    missing_month = 0
    for i, path in enumerate(files):
        passes, total = compute_pass_stats(path)
        rates.append(passes / total)
        total_passes += passes
        total_entries += total
        rid = replay_id_from_path(path)
        month = months.get(rid)
        if month is None:
            missing_month += 1
        else:
            by_month_passes[month] += passes
            by_month_entries[month] += total
            by_month_games[month] += 1
        if (i + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i + 1}/{len(files)}  ({(i + 1) / elapsed:.0f}/s)")

    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.2f}s ({len(files) / elapsed:.0f}/s)")
    if missing_month:
        print(f"  {missing_month} sampled replays had no row in DB (unexpected)")

    # Corpus-weighted rate is what the training sampler will actually see.
    # Per-game mean (in print_distribution) tells us per-game variance.
    print()
    print(
        f"Corpus-weighted pass-rate: {total_passes / total_entries:.4f}  "
        f"({total_passes:,} passes / {total_entries:,} entries)"
    )

    print()
    print_distribution("Per-game pass-rate", rates, bins=20)

    print()
    print("Pass-rate by month (corpus-weighted):")
    for month in sorted(by_month_passes):
        p = by_month_passes[month]
        e = by_month_entries[month]
        g = by_month_games[month]
        print(f"  {month}: games={g:>6}  rate={p / e:.4f}")


if __name__ == "__main__":
    main()
