#!/usr/bin/env -S uv run python
"""who-dies-next: model-free heuristic baselines + head-vs-baseline join.

The next_death aux head ([`6.13-12`]/[`6.13-13`]) is graded on whether it beats
the obvious leaderboard heuristics, not on its raw cross-entropy. The head dumps
carry its per-frame predictions (`next_elim_probs`/`target`/`alive_mask`/`dt`)
but no army/scoreboard, so the heuristics have to be computed separately from
sim ground truth and joined back on `(persp_val_index, frame_t)`.

This script does both halves:

  1. Stage-0 baseline table (model-free). Reproduces the run's val-split order
     so `persp_val_index` matches the dumps, walks each val perspective's frames
     the way the dataset does, and per frame computes which alive player two
     heuristics would name as the next victim:
       - lowest total army       — the "who's weakest" level signal (baseline #1)
       - most-negative army delta — the "attacked hardest" rate signal over a
         smoothing window K (baseline #2; per-tick delta is noisy)
     The true next-victim / alive-mask / horizon come from the exact target
     logic the head trains against (`next_death_target`), so the labels are
     identical to the dump's by construction (asserted at join time). The table
     is cached to `<sweep>/runs/baselines_stage0.npz` (run-independent: it's a
     function of the manifest + val split only).

  2. Head join + report. For each requested epoch dump, joins the head's argmax
     against the baseline table and prints aggregate top-1, the per-horizon
     (time-to-death) breakdown, and the disagreement read (when head and the
     lowest-army rule differ, which is right more often).

Usage:
    who_dies_next_baselines.py SWEEP_DIR [--manifest PATH] [--epochs 3,5]
    who_dies_next_baselines.py SWEEP_DIR --rebuild     # force baseline recompute

`SWEEP_DIR` is the sweep root (the dir holding `runs/` and `run_ids.txt`); the
report covers every cell in `run_ids.txt`. Point `--manifest` at the split JSON
the run trained on (default: the repo's `cloud_24k_sub_5k.json`).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from settings import TRAINING_DATA_DIR
from training.bc.eval.dump_compat import read_alive_mask
from training.bc.obs import canonical_slot_order
from training.bc.splits import load_manifest, samples_for_split
from training.bc.targets.elim_targets import ElimCtx, next_death_target, precompute_elim


# Reuse the model's bin edges only to size the winner sentinel in the elim
# precompute; the next_death target/horizon are invariant to the actual values
# (the sentinel just has to exceed every real death tick). These are the
# config-of-record edges, kept here so the script is self-contained.
EDGES = np.array([10, 20, 40, 80, 160, 320, 640], dtype=np.int64)
# Smoothing windows for the army-delta ("attacked hardest") baseline. K=1 is raw
# per-tick delta (noisy); the sweep tells us which window is most predictive.
DELTA_WINDOWS = (1, 5, 10, 25)
HORIZON_BUCKETS = ((0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 1 << 30))
DEFAULT_MANIFEST = TRAINING_DATA_DIR / "splits" / "cloud_24k_sub_5k.json"


def _army_totals_per_tick(sim: dict[str, np.ndarray]) -> np.ndarray:
    """`[T, 8]` total army per slot per tick (sum of armies over owned cells).

    Mirrors the leaderboard's army column: neutral / mountain cells are excluded
    by the `ownership == slot` mask, owned cities contribute their garrison.
    """
    own, arm = sim["ownership"], sim["armies"]  # [T, H, W]
    T = own.shape[0]
    tot = np.zeros((T, 8), dtype=np.int64)
    for s in range(8):
        tot[:, s] = np.where(own == s, arm, 0).reshape(T, -1).sum(1)
    return tot


def build_baseline_table(manifest_path: Path) -> dict[str, np.ndarray]:
    """Walk the val split and compute per-frame heuristic predictions + truth.

    Columns are parallel arrays over valid frames (winner-tail frames, which have
    no future death, are dropped — the head masks them too). Keyed for join by
    `(persp_val_index, frame_t)`, where `persp_val_index` is the perspective's
    position in the run's val-split order.
    """
    man = load_manifest(manifest_path)
    intermediate = Path(man["intermediate_root"])
    val = samples_for_split(man, "val", intermediate)
    print(f"val perspectives: {len(val)}")

    # persp_val_index = position in the val list; group by game to load each once.
    by_path: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for idx, (path, k) in enumerate(val):
        by_path[path].append((idx, k))

    cols: dict[str, list[int]] = defaultdict(list)
    big = np.iinfo(np.int64).max
    for path, persps in by_path.items():
        with np.load(path) as z:
            sim = {key: z[key] for key in z.files}
        with np.load(path.with_name(path.stem + ".meta.npz")) as z:
            meta = {key: z[key] for key in z.files}
        elim = ElimCtx(EDGES, *precompute_elim(sim, EDGES))
        army = _army_totals_per_tick(sim)  # [T, 8]
        T = sim["ownership"].shape[0]
        for persp_idx, k in persps:
            ps = int(meta["perspective_player_ids"][k])
            raw = np.asarray([ps, *canonical_slot_order(ps)[1:]], dtype=np.intp)
            elim_t = int(meta["elim_timestep"][k])
            end_t = T - 1 if elim_t == -1 else min(T - 1, elim_t)
            for t in range(end_t):
                victim, alive, dt = next_death_target(elim, list(raw), t)
                if victim < 0:  # winner tail: no future death, masked from loss
                    continue
                a = army[t, raw]  # army per canonical channel
                cols["persp_val_index"].append(persp_idx)
                cols["frame_t"].append(t)
                cols["victim"].append(victim)
                cols["dt"].append(dt)
                cols["low_army"].append(int(np.where(alive, a, big).argmin()))
                for K in DELTA_WINDOWS:
                    delta = np.where(alive, a - army[max(0, t - K), raw], big)
                    cols[f"delta{K}"].append(int(delta.argmin()))  # fastest loss
    table = {key: np.asarray(v) for key, v in cols.items()}
    print(f"frames: {len(table['victim'])}")
    return table


def report_stage0(table: dict[str, np.ndarray]) -> None:
    """Corpus marginals + aggregate baseline accuracies (the bar to beat)."""
    vic = table["victim"]
    dt = table["dt"]
    print("\n# Stage-0 corpus measurements (val split)")
    marg = " ".join(f"ch{c}:{(vic == c).mean() * 100:.1f}%" for c in range(8))
    print(f"  victim-channel marginal: {marg}")
    dist = {f"<{q}": round(float((dt < q).mean() * 100), 1) for q in (10, 25, 50, 100, 200)}
    print(f"  time-to-next-death CDF: {dist}  median={np.median(dt):.0f}")

    print("\n# Baseline top-1 next-victim accuracy (model-free)")
    print(f"  lowest total army : {(table['low_army'] == vic).mean() * 100:5.1f}%")
    for K in DELTA_WINDOWS:
        print(f"  army-delta K={K:<2}    : {(table[f'delta{K}'] == vic).mean() * 100:5.1f}%")
    # Combined-oracle upper bound: fraction where EITHER rule names the victim —
    # a ceiling on what a simple level+rate ensemble (and so a head with those
    # inputs) could reach.
    best_k = max(DELTA_WINDOWS, key=lambda K: (table[f"delta{K}"] == vic).mean())
    either = ((table["low_army"] == vic) | (table[f"delta{best_k}"] == vic)).mean()
    print(f"  ORACLE(low_army OR delta{best_k}) : {either * 100:5.1f}%  (simple-ensemble ceiling)")


def _index_table(table: dict[str, np.ndarray]) -> dict[tuple[int, int], int]:
    return {
        (int(pi), int(ft)): i
        for i, (pi, ft) in enumerate(zip(table["persp_val_index"], table["frame_t"]))
    }


def report_run(
    label: str,
    dump_path: Path,
    table: dict[str, np.ndarray],
    key_index: dict[tuple[int, int], int],
) -> None:
    """Join one epoch's head dump to the baselines and print the comparison."""
    z = np.load(dump_path)
    tgt = z["next_elim_target"]
    valid = tgt >= 0
    probs = np.where(read_alive_mask(z), z["next_elim_probs"].astype(np.float32), -1.0)
    head_pred = probs.argmax(1)
    pi, ft = z["persp_val_index"], z["frame_t"]

    rows = np.array([key_index[(int(pi[i]), int(ft[i]))] for i in np.where(valid)[0]])
    hp = head_pred[valid]
    vic = table["victim"][rows]
    dt = table["dt"][rows]
    low = table["low_army"][rows]
    delta = table[f"delta{max(DELTA_WINDOWS)}"][rows]
    assert (vic == tgt[valid]).all(), "join broken: victim label mismatch vs dump"

    print(f"\n### {label}  ({dump_path.parent.parent.name} / {dump_path.stem})  N={len(vic)}")
    print(f"  HEAD top-1        : {(hp == vic).mean() * 100:5.1f}%")
    print(f"  lowest total army : {(low == vic).mean() * 100:5.1f}%")
    print(f"  army-delta K={max(DELTA_WINDOWS):<2}    : {(delta == vic).mean() * 100:5.1f}%")
    disagree = hp != low
    print(
        f"  head vs lowest-army disagree on {disagree.mean() * 100:.0f}% of frames; "
        f"there head right {(hp[disagree] == vic[disagree]).mean() * 100:.1f}% "
        f"vs lowest-army right {(low[disagree] == vic[disagree]).mean() * 100:.1f}%"
    )
    print(f"  {'horizon dt':>12} {'frac':>6} {'head':>7} {'low_army':>9} {f'delta{max(DELTA_WINDOWS)}':>8}")
    for lo, hi in HORIZON_BUCKETS:
        sel = (dt >= lo) & (dt < hi)
        if not sel.any():
            continue
        name = f"{lo}-{hi if hi < (1 << 30) else 'inf'}"
        print(
            f"  {name:>12} {sel.mean() * 100:>5.1f}% "
            f"{(hp[sel] == vic[sel]).mean() * 100:>6.1f}% "
            f"{(low[sel] == vic[sel]).mean() * 100:>8.1f}% "
            f"{(delta[sel] == vic[sel]).mean() * 100:>7.1f}%"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", type=Path, help="sweep root (holds runs/ and run_ids.txt)")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="split manifest JSON")
    ap.add_argument("--epochs", default="best", help="comma-list of epochs, or 'all' / 'best' (default best=last present)")
    ap.add_argument("--rebuild", action="store_true", help="force recompute of the cached baseline table")
    args = ap.parse_args()

    cache = args.sweep_dir / "runs" / "baselines_stage0.npz"
    if cache.exists() and not args.rebuild:
        print(f"loading cached baselines: {cache}")
        table = dict(np.load(cache))
    else:
        table = build_baseline_table(args.manifest)
        np.savez(cache, **table)
        print(f"wrote {cache}")

    report_stage0(table)
    key_index = _index_table(table)

    run_ids = [line.split() for line in (args.sweep_dir / "run_ids.txt").read_text().split("\n") if line.strip()]
    for label, ts in run_ids:
        analysis_dir = args.sweep_dir / "runs" / ts / "analysis"
        dumps = sorted(analysis_dir.glob("stratified_val_epoch_*.npz"))
        if not dumps:
            continue
        present = [int(p.stem.rsplit("_", 1)[1]) for p in dumps]
        if args.epochs == "all":
            want = present
        elif args.epochs == "best":
            want = [max(present)]
        else:
            want = [int(e) for e in args.epochs.split(",") if int(e) in present]
        for ep in want:
            report_run(label, analysis_dir / f"stratified_val_epoch_{ep:03d}.npz", table, key_index)


if __name__ == "__main__":
    main()
