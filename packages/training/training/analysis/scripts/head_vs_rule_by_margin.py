#!/usr/bin/env -S uv run python
"""who-dies-next: head vs lowest-army rule, stratified by close-call margin.

[`6.13-15`]/[`6.14-1`] established that the next_death head trails the lowest-army
rule in the imminent (dt<10) regime, stratified by *time-to-death*. This script
adds a different cut — stratify by the **bottom-two army margin** (the gap between
the two lowest-army alive players) — to ask *where* the head loses to the rule:

  - If the head merely reproduces the rule, head==rule agreement is ~uniform and
    accuracies track across margin bins.
  - If the head's deficit concentrates at small margin, close calls are the wall.
  - The actual finding: the gap is largest in the *mid*-margin regime — where army
    cleanly names the victim but the head still misses — which points at the head
    not using the army ranking, not at irreducibly-hard close calls.

It is a thin consumer of `fq`: build the ELIM_HEAD_DEBUG table (army_sim, the rule,
and the margin as columns), `join_dump` the head's predictions onto it on the
full-split key, then slice with numpy. The val-split walk, the single-source army
decode, and the join-integrity check are all the toolkit's job — compare the
hand-rolled equivalent in `who_dies_next_baselines.py`.

Frames are clustered by game, so the confidence intervals come from a cluster
bootstrap over `game_id` (resampling frames would understate the uncertainty).

Usage:
    head_vs_rule_by_margin.py DUMP [--manifest PATH] [--max-games N] [--dt-max 10]

`DUMP` is a `stratified_val_epoch_*.npz` from the run whose split `--manifest`
describes (they must match, or the join key misaligns — `join_dump` asserts on it).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from settings import TRAINING_DATA_DIR
from training.analysis.families.elim_head_debug import ELIM_HEAD_DEBUG
from training.analysis.fq.frame_table import (
    GROUND_TRUTH_OBS_CFG,
    FrameTable,
    build_frame_table,
    join_dump,
    select,
)
from training.bc.splits import load_manifest, samples_for_split


DEFAULT_MANIFEST = TRAINING_DATA_DIR / "splits" / "cloud_24k_sub_11k.json"
MARGIN_EDGES = (0, 10, 25, 50, 100, 250, np.inf)


def build_imminent(dump: Path, manifest: Path, max_games: int | None, dt_max: int) -> FrameTable:
    """Build the ELIM_HEAD_DEBUG table on the manifest's val split, join the dump's
    head predictions onto it, and return the imminent (0 <= dt < dt_max) slice with
    a real next victim."""
    man = load_manifest(manifest)
    samples = samples_for_split(man, "val", Path(man["intermediate_root"]))
    t = build_frame_table(ELIM_HEAD_DEBUG, samples, GROUND_TRUTH_OBS_CFG, max_games)
    j = join_dump(t, dump, ELIM_HEAD_DEBUG.truth_map)

    imm = select(j, (j.cols["dt"] >= 0) & (j.cols["dt"] < dt_max))
    return select(imm, imm.cols["victim"] >= 0)


def predictions(j: FrameTable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (true victim, rule victim, head victim) per frame, all in the canonical
    [perspective, *opp] channel space. The head argmaxes over *alive* channels only —
    the rule already does (the lowest-army column), and the head puts ~0 mass on dead
    slots, so this just makes the comparison provably apples-to-apples."""
    victim = j.cols["victim"]
    rule = j.cols["low_army_victim"]
    head = np.where(j.cols["alive"], j.cols["next_elim_probs"], -np.inf).argmax(1)
    return victim, rule, head


def cluster_bootstrap_ci(
    correct: np.ndarray, game_id: np.ndarray, n_boot: int, seed: int = 0
) -> tuple[float, float]:
    """95% CI for the mean of a per-frame boolean, resampling whole games with
    replacement (frames within a game are correlated, so the cluster is the game)."""
    games = np.unique(game_id)
    rows_by_game = [np.where(game_id == g)[0] for g in games]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(games), len(games))
        rows = np.concatenate([rows_by_game[i] for i in pick])
        means[b] = correct[rows].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(j: FrameTable, n_boot: int, dt_max: int) -> None:
    victim, rule, head = predictions(j)
    margin = j.cols["margin"]
    rule_ok, head_ok = (rule == victim), (head == victim)

    print(f"\nimminent (dt<{dt_max}) victim frames: {victim.size}  /  {j.n_games} games\n")
    print("  margin bin            n      rule    head     gap   head==rule")
    for lo, hi in zip(MARGIN_EDGES[:-1], MARGIN_EDGES[1:], strict=True):
        b = (margin >= lo) & (margin < hi)
        n = int(b.sum())
        if not n:
            continue
        r, h = rule_ok[b].mean() * 100, head_ok[b].mean() * 100
        hi_lbl = "inf" if np.isinf(hi) else f"{hi:.0f}"
        print(
            f"  [{lo:>4.0f},{hi_lbl:>4})    {n:7d}   {r:4.1f}%   {h:4.1f}%   "
            f"{r - h:+5.1f}    {(head[b] == rule[b]).mean() * 100:5.1f}%"
        )
    r, h = rule_ok.mean() * 100, head_ok.mean() * 100
    print(
        f"  overall          {victim.size:7d}   {r:4.1f}%   {h:4.1f}%   {r - h:+5.1f}    "
        f"{(head == rule).mean() * 100:5.1f}%"
    )

    if n_boot:
        rl, ru = cluster_bootstrap_ci(rule_ok, j.game_id, n_boot)
        hl, hu = cluster_bootstrap_ci(head_ok, j.game_id, n_boot)
        gl, gu = cluster_bootstrap_ci(rule_ok.astype(float) - head_ok, j.game_id, n_boot)
        print(f"\ncluster-bootstrap 95% CIs ({n_boot} resamples over {j.n_games} games):")
        print(f"  rule        {r:4.1f}%   CI [{rl * 100:.1f}, {ru * 100:.1f}]")
        print(f"  head        {h:4.1f}%   CI [{hl * 100:.1f}, {hu * 100:.1f}]")
        print(f"  rule - head {r - h:+5.1f}   CI [{gl * 100:.1f}, {gu * 100:.1f}]")

    dis = head != rule
    print(
        f"\ndisagree {dis.mean() * 100:.1f}%  ->  head right {head_ok[dis].mean() * 100:.1f}%   "
        f"rule right {rule_ok[dis].mean() * 100:.1f}%"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dump", type=Path, help="stratified_val_epoch_*.npz from the run")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--max-games", type=int, default=None, help="cap to first N games")
    ap.add_argument("--dt-max", type=int, default=10, help="imminent horizon (ticks)")
    ap.add_argument("--bootstrap", type=int, default=2000, help="cluster resamples; 0 to skip")
    args = ap.parse_args()

    j = build_imminent(args.dump, args.manifest, args.max_games, args.dt_max)
    report(j, args.bootstrap, args.dt_max)


if __name__ == "__main__":
    main()
