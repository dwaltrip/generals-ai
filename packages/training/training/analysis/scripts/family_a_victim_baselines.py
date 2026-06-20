#!/usr/bin/env -S uv run python
"""Family-A who-dies-next baselines: model-free heuristics + head-vs-baseline join.

The next_death aux head is graded against the obvious always-visible-leaderboard
heuristics ([`6.18-9`] §2: the lowest-army rule is a *floor*, not a target; the
head must beat it AND win the frames where it disagrees — [`6.19-2`]). This script
measures those heuristics over the val split and joins each epoch's head dump onto
them on the full-split `(persp_val_index, frame_t)` key.

The heuristics are two families:

  - **Order statistics** over current army: `level` (lowest army — the standing
    floor, ~47% top-1), plus `2nd_lowest` / `3rd_lowest` (controls that bound how
    much the OR-an-oracle gains come from just looking past the single weakest).
  - **Rate constructions** — "who's being attacked hardest" over a trailing window
    K: army-loss read as an `endpoint` delta (army[t] − army[t−K]) or as a
    reduction over the trailing per-tick deltas (`worst_tick`, `topN`, percentiles),
    each in an `absolute` and a `relative` (loss-as-a-fraction) flavor.

Each heuristic names a predicted victim = argmin-over-alive of its score column.
Scoring is consumer-side numpy: hard top-1; the soft "is the pick among the next-k
to actually die" (k=1,2,3); the `oracle(level OR heuristic)` ceiling; and the exact
closed-form `(level OR random)` bar a heuristic must clear to add anything over the
floor + a coin flip.

Built on fq: ONE `FrameTable` over the val split carries the army signal, every
heuristic's score column, and the next_death truth (victim/dt/alive) from the same
target path the dumps use; `join_dump` attaches each dump's head predictions and
cross-checks the shared truth columns. The score columns lean on two fq derivers —
`windowed_per_game` (trailing-K reductions) and `per_game` (the endpoint delta).

Usage:
    family_a_victim_baselines.py SWEEP_DIR [--manifest PATH] [--epochs best|all|CSV]
                                 [--max-games N]

`SWEEP_DIR` is the sweep root (holds `runs/` and `run_ids.txt`); the report covers
every cell in `run_ids.txt`, skipping any whose dumps carry no `next_elim_*` columns
(an elim-off control run). Point `--manifest` at the split the runs trained on
(default `cloud_24k_sub_5k.json`, the canonical reference split); a mismatch
misaligns the join key (asserted by `join_dump`).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from settings import TRAINING_DATA_DIR
from training.analysis.families.army_derivers import (
    army_totals_per_tick,
    kth_lowest_alive,
)
from training.analysis.fq.derivers import per_game, windowed_per_game
from training.analysis.fq.frame_table import (
    GROUND_TRUTH_OBS_CFG,
    FrameTable,
    FrameTableSpec,
    build_frame_table,
    join_dump,
    select,
)
from training.bc.splits import load_manifest, samples_for_split
from training.bc.targets.elim_targets import precompute_elim


DEFAULT_MANIFEST = TRAINING_DATA_DIR / "splits" / "cloud_24k_sub_5k.json"
# Trailing-window lengths for the rate constructions (ticks). K=25 is the
# config-of-record; the list is the menu's tuning axis.
WINDOWS = (5, 10, 25)
HORIZON_BUCKETS = ((0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 1 << 30))

# --- The army signal. ONE pointer, so swapping army->land is a single edit. ---
# NOTE: land is the next intended consumer ([`6.18-9`] §2.5/§3.B), but it is NOT a
# 1:1 reuse: land deltas track territory flips on a different cadence than army
# (army moves with combat losses tick-to-tick; land moves in captures), so land
# will likely want land-SPECIFIC constructions/windows, not these army ones. Build
# army now; do not implement land here.
SIGNAL_PER_TICK = army_totals_per_tick


def _abs_delta_per_tick(sim: dict[str, np.ndarray]) -> np.ndarray:
    """`[T, 8]` per-tick change in the signal: row t = signal[t] − signal[t−1].
    Row 0 is NaN (no prior tick), so the trailing-window reductions reduce over only
    real deltas at the start of a game (the `windowed_per_game` NaN contract)."""
    s = SIGNAL_PER_TICK(sim).astype(np.float64)
    d = np.full_like(s, np.nan)
    d[1:] = np.diff(s, axis=0)
    return d


def _rel_delta_per_tick(sim: dict[str, np.ndarray]) -> np.ndarray:
    """`[T, 8]` per-tick *relative* change: each tick's delta as a fraction of the
    signal just before that tick (delta normalized PER TICK, before any windowing).
    Guarded to 0 where the prior value is non-positive; row 0 is NaN."""
    s = SIGNAL_PER_TICK(sim).astype(np.float64)
    out = np.full_like(s, np.nan)
    prev = s[:-1]
    out[1:] = np.where(prev > 0, np.diff(s, axis=0) / np.maximum(prev, 1e-9), 0.0)
    return out


# --- NaN-aware window reductions (lower score => more likely victim). ---
# These reduce a `[T, K, 8]` trailing-window tensor over axis 1 (the K ticks),
# NaN-aware so short start-of-game windows fold over only their real values. They
# sort once and read order statistics off the sorted window — `np.nanpercentile`
# is ~an order of magnitude slower, and with 36 columns built per game that
# dominates the table build.


def _sort_window(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort the window ascending along axis 1 (NaN pads land last) and return the
    sorted tensor plus the per-(tick, slot) count of real (non-NaN) values."""
    Ws = np.sort(W, axis=1)                       # [T, K, 8]; NaN sorts to the end
    n = (~np.isnan(W)).sum(axis=1)                # [T, 8] real count per window
    return Ws, n


def _r_worst_tick(W: np.ndarray) -> np.ndarray:
    """The single worst (most negative) per-tick delta in the window."""
    Ws, _ = _sort_window(W)
    return Ws[:, 0, :]                            # smallest real value (NaN sorts last)


def _r_topn_mean(n: int) -> Callable[[np.ndarray], np.ndarray]:
    """Mean of the `n` most negative per-tick deltas (fewer where the window is
    shorter than `n`). The first `n` sorted slots are the real smallest values;
    nanmean averages over however many of them are real."""

    def reduce(W: np.ndarray) -> np.ndarray:
        Ws, _ = _sort_window(W)
        return np.nanmean(Ws[:, :n, :], axis=1)

    return reduce


def _r_pct(p: float) -> Callable[[np.ndarray], np.ndarray]:
    """The p-th percentile of the window's per-tick deltas (e.g. p=10 => the worst
    decile). Computed off the per-window sort with numpy's default 'linear'
    interpolation over the real count, matching `np.nanpercentile`."""

    def reduce(W: np.ndarray) -> np.ndarray:
        Ws, cnt = _sort_window(W)                 # [T, K, 8], [T, 8]
        cnt = np.maximum(cnt, 1)
        pos = (cnt - 1) * (p / 100.0)             # virtual index into the sorted reals
        lo = np.floor(pos).astype(np.intp)
        frac = pos - lo
        # Gather the bracketing sorted values per (tick, slot) along axis 1.
        lo_v = np.take_along_axis(Ws, lo[:, None, :], axis=1)[:, 0, :]
        hi_v = np.take_along_axis(Ws, (lo + 1).clip(max=Ws.shape[1] - 1)[:, None, :], axis=1)[:, 0, :]
        # Where the position is exact (frac == 0) keep lo_v as-is: hi_v there can be a
        # NaN pad (count == 1, or the last real index), and 0 * NaN would poison it.
        return np.where(frac > 0, lo_v + frac * (hi_v - lo_v), lo_v)

    return reduce


@dataclass(frozen=True)
class Heuristic:
    """One model-free victim-namer. `kind` selects how `predict` reads the table:

      - "order": k-th-lowest-army order statistic (`k` set).
      - "score": argmin-over-alive of the named score column (`score_col` set).

    Adding a construction = appending one entry here + (for "score") wiring its
    column in `build_score_derivers`.
    """

    name: str
    kind: str
    k: int = 0
    score_col: str = ""


# The heuristic menu, declarative. Order statistics + the rate constructions
# {endpoint, worst_tick, top3, top5, pct90, pct95} x {absolute, relative} x windows.
def build_menu() -> list[Heuristic]:
    menu = [
        Heuristic("level", "order", k=0),
        Heuristic("2nd_lowest", "order", k=1),
        Heuristic("3rd_lowest", "order", k=2),
    ]
    for K in WINDOWS:
        for flavor in ("abs", "rel"):
            for stat in ("endpoint", "worst_tick", "top3", "top5", "pct90", "pct95"):
                menu.append(Heuristic(f"{flavor}_{stat}_{K}", "score", score_col=f"{flavor}_{stat}_{K}"))
    return menu


def build_score_derivers() -> list:
    """One per-game deriver per rate-construction score column. Endpoint is a plain
    `per_game` delta (army[t] − army[t−K]); the per-tick reductions are
    `windowed_per_game` over the absolute / relative per-tick delta series."""
    derivers = []
    reducers = {
        "worst_tick": _r_worst_tick,
        "top3": _r_topn_mean(3),
        "top5": _r_topn_mean(5),
        "pct90": _r_pct(10),   # worst decile of loss
        "pct95": _r_pct(5),    # worst 5%
    }
    for K in WINDOWS:
        # endpoint: window-START to now. Absolute = army[t] - army[t-K]; relative =
        # that delta / army[t-K] (normalized at the window START, unlike the
        # per-tick constructions which normalize each tick before reducing).
        derivers.append(_endpoint_deriver(f"abs_endpoint_{K}", K, relative=False))
        derivers.append(_endpoint_deriver(f"rel_endpoint_{K}", K, relative=True))
        for stat, reduce in reducers.items():
            derivers.append(
                windowed_per_game(f"abs_{stat}_{K}", True, _abs_delta_per_tick, reduce, K)
            )
            derivers.append(
                windowed_per_game(f"rel_{stat}_{K}", True, _rel_delta_per_tick, reduce, K)
            )
    return derivers


def _endpoint_deriver(name: str, K: int, relative: bool):
    """A `per_game` deriver: score[t] = signal[t] − signal[clamp(t−K, 0)], optionally
    divided by signal at the window start (the relative-endpoint denominator)."""

    def prepare(sim: dict[str, np.ndarray]) -> np.ndarray:
        s = SIGNAL_PER_TICK(sim).astype(np.float64)
        T = s.shape[0]
        start = np.maximum(np.arange(T) - K, 0)
        delta = s - s[start]
        if relative:
            return delta / np.maximum(s[start], 1e-9)
        return delta

    return per_game(name, True, prepare, lambda state, f: state[f.t, f.raw_order])


# A game-independent stand-in for "not a real future death", larger than any real
# death tick. The per-game winner sentinel (T+1) and phantom slots (-1) both map to
# it, so the soft scorer's "real death" filter is a single `death < NO_DEATH` test
# with no per-game sentinel to thread.
NO_DEATH = np.int64(1 << 30)


def _death_tick_deriver():
    """A `per_game` deriver: each channel's elimination tick. Real deaths keep their
    tick; the winner sentinel and phantom slots both collapse to `NO_DEATH` so the
    soft scorer excludes them with one comparison."""

    def prepare(sim: dict[str, np.ndarray]) -> np.ndarray:
        death_by_slot, _removal, _is_real, sentinel = precompute_elim(sim, None)
        real = (death_by_slot >= 0) & (death_by_slot < sentinel)
        return np.where(real, death_by_slot, NO_DEATH)

    return per_game(
        "death_tick", True, prepare, lambda state, f: state[f.raw_order]
    )


def build_spec() -> FrameTableSpec:
    """The Family-A table: army signal + every score column + next_death truth.

    `victim`/`dt`/`alive` are emitted from the dataset's `next_death` path — the same
    truth the dumps carry — so `truth_map` lets `join_dump` assert agreement on the
    overlap (the persp_val_index cross-check)."""
    army = per_game(
        "army_sim", True, SIGNAL_PER_TICK,
        lambda totals, f: totals[f.t, f.raw_order].astype(float),
    )
    return FrameTableSpec(
        name="family_a_victim",
        dataset_kwargs=dict(
            elim_head_variant="next_death", include_frame_info=True, unsafe_attach_sim_frame=True
        ),
        emit_cols={"alive_mask": "alive", "next_elim_target": "victim", "next_elim_dt": "dt"},
        derivers=[army, _death_tick_deriver(), *build_score_derivers()],
        # Dumps write the canonical `alive_mask`; the per-frame next_death target and
        # horizon are the shared truth the join cross-checks.
        truth_map={"victim": "next_elim_target", "dt": "next_elim_dt", "alive": "alive_mask"},
    )


def predict(t: FrameTable, h: Heuristic) -> np.ndarray:
    """`[N]` predicted victim channel for one heuristic over the table's frames."""
    alive = t.cols["alive"].astype(bool)
    if h.kind == "order":
        return kth_lowest_alive(t.cols["army_sim"], alive, h.k)
    score = np.where(alive, t.cols[h.score_col], np.inf)
    return score.argmin(axis=1)


def soft_among_next_k(t: FrameTable, pred: np.ndarray) -> np.ndarray:
    """`[N, 3]` bool: is `pred` among the next 1/2/3 alive channels to actually die?

    Orders the alive channels by their real death tick (winner/phantom excluded) and
    asks whether the pick lands in the soonest-k. [`6.18-9`] §3.C's soft target: near-
    simultaneous deaths make the hard one-hot an irreducible coin flip, so credit a
    pick that names ANY of the next few victims."""
    alive = t.cols["alive"].astype(bool)
    death = t.cols["death_tick"]
    real = alive & (death < NO_DEATH)
    big = np.iinfo(np.int64).max
    # Rank each channel by death tick among the real future deaths (rank 0 = soonest).
    # Non-real channels sort last, so their rank is >= n_real and never < k below.
    order = np.argsort(np.where(real, death, big), axis=1, kind="stable")  # [N, 8]
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(8)[None, :], axis=1)          # channel -> rank
    pred_rank = np.take_along_axis(rank, pred[:, None], axis=1)[:, 0]      # [N]
    pred_real = np.take_along_axis(real, pred[:, None], axis=1)[:, 0]      # [N]
    out = np.zeros((pred.size, 3), dtype=bool)
    for ki in range(3):
        # The pick lands among the next (ki+1) to die iff it's a real death ranked
        # within the soonest (ki+1). Frames with fewer real deaths than k just can't
        # exceed their own count, so next-2/next-3 collapse to next-1 there.
        out[:, ki] = pred_real & (pred_rank <= ki)
    return out


def report_marginals(t: FrameTable) -> None:
    vic, dt = t.cols["victim"], t.cols["dt"]
    print(f"\n# Corpus measurements (val split): {vic.size} frames / {t.n_games} games")
    marg = " ".join(f"ch{c}:{(vic == c).mean() * 100:.1f}%" for c in range(8))
    print(f"  victim-channel marginal: {marg}")
    cdf = {f"<{q}": round(float((dt < q).mean() * 100), 1) for q in (10, 25, 50, 100, 200)}
    print(f"  time-to-next-death CDF: {cdf}  median={np.median(dt):.0f}")


def report_baselines(t: FrameTable, menu: list[Heuristic]) -> None:
    """Standalone top-1, oracle(level OR ...), and the (level OR random) bar."""
    vic = t.cols["victim"]
    n_alive = t.cols["alive"].astype(bool).sum(axis=1)
    level_pred = predict(t, menu[0])               # menu[0] is `level` by construction
    level_ok = level_pred == vic
    # Exact expected oracle(level OR random): on a level miss, a uniform random pick
    # among the alive players is right with prob 1/n_alive.
    rand_bar = (level_ok + (~level_ok) / np.maximum(n_alive, 1)).mean() * 100

    rows = []
    for h in menu:
        pred = predict(t, h)
        ok = (pred == vic).mean() * 100
        oracle = (level_ok | (pred == vic)).mean() * 100
        rows.append((h.name, ok, oracle))

    print(f"\n# Model-free baselines  (level OR random bar = {rand_bar:.1f}%)")
    print(f"  {'heuristic':>16}  {'standalone':>10}  {'oracle(lvl OR)':>14}")
    for name, ok, oracle in sorted(rows, key=lambda r: -r[2]):
        flag = "  BEATS-BAR" if oracle > rand_bar else ""
        print(f"  {name:>16}: {ok:9.1f}%  {oracle:13.1f}%{flag}")


def report_soft(t: FrameTable, menu: list[Heuristic]) -> None:
    """Soft 'among next-k to die' for a few representative heuristics."""
    want = ["level", "2nd_lowest", "abs_endpoint_25", "rel_endpoint_25", "abs_pct90_25", "rel_pct90_25"]
    by_name = {h.name: h for h in menu}
    print("\n# Soft 'pick among next-k to die'  (next-1 == hard top-1)")
    print(f"  {'heuristic':>16}  {'next-1':>7} {'next-2':>7} {'next-3':>7}")
    for name in want:
        if name not in by_name:
            continue
        soft = soft_among_next_k(t, predict(t, by_name[name]))
        s = soft.mean(axis=0) * 100
        print(f"  {name:>16}: {s[0]:6.1f}% {s[1]:6.1f}% {s[2]:6.1f}%")


def report_run(label: str, ts: str, dump: Path, t: FrameTable, spec: FrameTableSpec) -> None:
    """Join one epoch dump and report head top-1, the disagreement read, and the
    per-horizon breakdown vs the lowest-army rule."""
    j = join_dump(t, dump, spec.truth_map)
    vic = j.cols["victim"]
    alive = j.cols["alive"].astype(bool)
    head = np.where(alive, j.cols["next_elim_probs"].astype(np.float32), -np.inf).argmax(1)
    level = kth_lowest_alive(j.cols["army_sim"], alive, 0)
    head_ok, level_ok = head == vic, level == vic

    print(f"\n### {label}  ({ts} / {dump.stem})  N={vic.size}")
    print(f"  HEAD top-1        : {head_ok.mean() * 100:5.1f}%")
    print(f"  level (low army)  : {level_ok.mean() * 100:5.1f}%")
    dis = head != level
    print(
        f"  head vs level disagree on {dis.mean() * 100:.0f}% of frames; there "
        f"head right {head_ok[dis].mean() * 100:.1f}% vs level right {level_ok[dis].mean() * 100:.1f}%"
    )
    dt = j.cols["dt"]
    print(f"  {'horizon dt':>12} {'frac':>6} {'head':>7} {'level':>7}")
    for lo, hi in HORIZON_BUCKETS:
        sel = (dt >= lo) & (dt < hi)
        if not sel.any():
            continue
        name = f"{lo}-{hi if hi < (1 << 30) else 'inf'}"
        print(
            f"  {name:>12} {sel.mean() * 100:>5.1f}% "
            f"{head_ok[sel].mean() * 100:>6.1f}% {level_ok[sel].mean() * 100:>6.1f}%"
        )


def epoch_dumps(analysis_dir: Path, epochs: str) -> list[Path]:
    dumps = sorted(analysis_dir.glob("stratified_val_epoch_*.npz"))
    if not dumps:
        return []
    present = [int(p.stem.rsplit("_", 1)[1]) for p in dumps]
    if epochs == "all":
        want = present
    elif epochs == "best":
        want = [max(present)]
    else:
        want = [int(e) for e in epochs.split(",") if int(e) in present]
    return [analysis_dir / f"stratified_val_epoch_{ep:03d}.npz" for ep in want]


def has_elim_columns(dump: Path) -> bool:
    """True iff the dump carries the next_death head outputs. An elim-off control run
    dumps only value/policy/placement, so its `next_elim_*` columns are absent and the
    join would fail — detect and skip it instead."""
    with np.load(dump) as z:
        return "next_elim_probs" in z.files and "next_elim_target" in z.files


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("sweep_dir", type=Path, help="sweep root (holds runs/ and run_ids.txt)")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="split manifest JSON")
    ap.add_argument(
        "--epochs", default="best", help="comma-list of epochs, or 'all' / 'best' (default: best=last)"
    )
    ap.add_argument("--max-games", type=int, default=None, help="cap to first N games (fast iteration)")
    args = ap.parse_args()

    spec = build_spec()
    man = load_manifest(args.manifest)
    samples = samples_for_split(man, "val", Path(man["intermediate_root"]))
    print(f"building Family-A table: {len(samples)} val perspectives "
          f"({'all games' if args.max_games is None else f'first {args.max_games} games'})")
    t = build_frame_table(spec, samples, GROUND_TRUTH_OBS_CFG, args.max_games)

    # Hard-victim frames (a real next death exists; the winner tail is masked) from
    # t>=1 — at t=0 the rate constructions have no trailing delta (an all-NaN window),
    # so the heuristics are degenerate; the references drop it too.
    t = select(t, (t.cols["victim"] >= 0) & (t.frame_t >= 1))

    menu = build_menu()
    report_marginals(t)
    report_baselines(t, menu)
    report_soft(t, menu)

    run_ids = [
        line.split()
        for line in (args.sweep_dir / "run_ids.txt").read_text().split("\n")
        if line.strip()
    ]
    for label, ts in run_ids:
        analysis_dir = args.sweep_dir / "runs" / ts / "analysis"
        dumps = epoch_dumps(analysis_dir, args.epochs)
        if not dumps:
            print(f"\n### {label} ({ts}): no epoch dumps — skipping")
            continue
        if not has_elim_columns(dumps[0]):
            print(f"\n### {label} ({ts}): no next_elim_* columns (elim-off run) — skipping")
            continue
        for dump in dumps:
            report_run(label, ts, dump, t, spec)


if __name__ == "__main__":
    main()
