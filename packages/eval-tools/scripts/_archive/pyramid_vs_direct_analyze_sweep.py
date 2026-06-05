#!/usr/bin/env -S uv run python
"""Analyze a multi-matchup eval sweep produced by eval_sweep.sh.

Usage:
    ./packages/eval-tools/scripts/analyze_sweep.py [sweep_dir]

With no arg, uses the most recent data/eval-runs/sweep-* dir.

Produces:
  - Win-rate table (cross-kind matchups)  + 95% Wilson CI + binomial p
  - Self-vs-self balance check
  - Per-policy diagnostic averages (mean of per-game means), split by win/loss
  - Pyramid value-head trajectory check (memorization vs calibration)
  - Pyramid win rate as a function of temperature
"""

from __future__ import annotations

import argparse
import json
import math
from math import comb
from pathlib import Path
import statistics
from typing import Any


def policy_kind(spec: str) -> str:
    return "pyramid" if "value_head_variant=pyramid" in spec else "direct"


def policy_temperature(spec: str) -> str:
    parts = spec.split(":", 2)
    if len(parts) < 3:
        return "?"
    for token in parts[2].split(","):
        if token.startswith("temperature="):
            return token.split("=", 1)[1]
    return "1.0"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfw = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - halfw), min(1.0, center + halfw)


def binomial_pvalue_ge(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k | n, p) — one-sided."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def find_latest_sweep() -> Path:
    candidates = sorted(Path("data/eval-runs").glob("sweep-*"))
    if not candidates:
        raise SystemExit("No sweep dirs found under data/eval-runs/")
    return candidates[-1]


def load_matchup(run_dir: Path) -> dict:
    with (run_dir / "config.json").open() as f:
        cfg = json.load(f)
    specs = cfg["policy_specs"]
    rows = []
    with (run_dir / "results.jsonl").open() as f:
        for line in f:
            rows.append(json.loads(line))
    kinds = [policy_kind(s) for s in specs]
    temps = [policy_temperature(s) for s in specs]
    temp = temps[0] if all(t == temps[0] for t in temps) else "mixed"
    label = (
        f"{kinds[0]} vs {kinds[0]} @ temp={temp}"
        if kinds[0] == kinds[1]
        else f"{kinds[0]} vs {kinds[1]} @ temp={temp}"
    )
    return {
        "dir": run_dir,
        "specs": specs,
        "kinds": kinds,
        "temp": temp,
        "games": rows,
        "label": label,
        "is_self": kinds[0] == kinds[1],
    }


def winner_policy_index(game: dict) -> int | None:
    if game["winner"] is None:
        return None
    return game["slot_map"][game["winner"]]


def policy_slot_in_game(game: dict, policy_idx: int) -> int:
    """Which slot was this original policy occupying in this game?"""
    return game["slot_map"].index(policy_idx)


def get_diag_for_policy(game: dict, policy_idx: int) -> dict | None:
    slot = policy_slot_in_game(game, policy_idx)
    diags = game.get("metrics", {}).get("policy_diagnostics") or []
    for d in diags:
        if d.get("player") == slot:
            return d
    return None


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def analyze_winrates(cross: list[dict]) -> None:
    if not cross:
        return
    print_header("Win rates — cross-kind matchups")
    print()
    hdr = (
        f"{'Matchup':<38} {'N':>3} {'P0kind':<8} {'P1kind':<8} "
        f"{'P0w':>4} {'P1w':>4} {'Drw':>4} {'P1 rate':>8} {'95% CI':>15} {'p≥(½)':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for m in cross:
        n = len(m["games"])
        p0w = sum(1 for g in m["games"] if winner_policy_index(g) == 0)
        p1w = sum(1 for g in m["games"] if winner_policy_index(g) == 1)
        draws = n - p0w - p1w
        decided = p0w + p1w
        rate = p1w / decided if decided else float("nan")
        lo, hi = wilson_ci(p1w, decided) if decided else (0.0, 0.0)
        p_val = binomial_pvalue_ge(p1w, decided) if decided else float("nan")
        ci = f"{lo*100:4.1f}%-{hi*100:4.1f}%"
        print(
            f"{m['label']:<38} {n:>3} {m['kinds'][0]:<8} {m['kinds'][1]:<8} "
            f"{p0w:>4} {p1w:>4} {draws:>4} {rate*100:>7.1f}% {ci:>15} {p_val:>8.3f}"
        )


def analyze_self_vs_self(self_pairs: list[dict]) -> None:
    if not self_pairs:
        return
    print_header("Self-vs-self — noise floor (expect ~50/50)")
    print()
    hdr = (
        f"{'Matchup':<42} {'N':>3} {'Slot0w':>7} {'Slot1w':>7} {'Drw':>4} "
        f"{'Slot1 rate':>10} {'95% CI':>15}"
    )
    print(hdr)
    print("-" * len(hdr))
    for m in self_pairs:
        n = len(m["games"])
        s0w = sum(1 for g in m["games"] if g["winner"] == 0)
        s1w = sum(1 for g in m["games"] if g["winner"] == 1)
        draws = n - s0w - s1w
        decided = s0w + s1w
        rate = s1w / decided if decided else float("nan")
        lo, hi = wilson_ci(s1w, decided) if decided else (0.0, 0.0)
        ci = f"{lo*100:4.1f}%-{hi*100:4.1f}%"
        balance_flag = ""
        if decided and (rate < 0.30 or rate > 0.70):
            balance_flag = "  ← noise floor concerning"
        print(
            f"{m['label']:<42} {n:>3} {s0w:>7} {s1w:>7} {draws:>4} "
            f"{rate*100:>9.1f}% {ci:>15}{balance_flag}"
        )


def diag_field_mean(games: list[dict], policy_idx: int, path: list[str]) -> float:
    """Mean of nested diagnostic field across games (mean-of-means)."""
    vals = []
    for g in games:
        d = get_diag_for_policy(g, policy_idx)
        if d is None:
            continue
        node: Any = d  # untyped nested-dict walk; leaf validated by the try/except
        try:
            for k in path:
                node = node[k]
            vals.append(float(node))
        except (KeyError, TypeError):
            continue
    return statistics.mean(vals) if vals else float("nan")


def analyze_diagnostics(matchups: list[dict]) -> None:
    print_header("Per-policy diagnostics (mean of per-game means)")
    fields = [
        ("value.exp_placement",  ["value", "exp_placement", "mean"]),
        ("value.top_prob",       ["value", "top_prob", "mean"]),
        ("value.entropy",        ["value", "entropy", "mean"]),
        ("policy.top1_prob",     ["policy", "top1_prob", "mean"]),
        ("policy.entropy",       ["policy", "entropy", "mean"]),
        ("pass_prob",            ["pass_prob", "mean"]),
    ]
    for m in matchups:
        print()
        print(f"  {m['label']}  ({len(m['games'])} games)")
        for pi in (0, 1):
            kind = m["kinds"][pi]
            won_games = [g for g in m["games"] if winner_policy_index(g) == pi]
            lost_games = [g for g in m["games"]
                          if winner_policy_index(g) is not None
                          and winner_policy_index(g) != pi]
            print(f"    P{pi} ({kind}):  won {len(won_games)}, lost {len(lost_games)}")
            for label, path in fields:
                all_mean = diag_field_mean(m["games"], pi, path)
                w_mean = diag_field_mean(won_games, pi, path)
                l_mean = diag_field_mean(lost_games, pi, path)
                print(
                    f"      {label:<22} "
                    f"all={all_mean:7.3f}  won={w_mean:7.3f}  lost={l_mean:7.3f}"
                )


def memorization_check(cross: list[dict]) -> None:
    print_header("Pyramid value-head trajectory — memorization vs calibration")
    print()
    print("  exp_placement scale: 1=most confident win, 8=most confident loss.")
    print("  If pyramid stays near 1.0 even in losses → memorization.")
    print("  If pyramid drifts toward higher placement during losses → calibrated.")
    pyramid_cross = [m for m in cross if "pyramid" in m["kinds"]]
    for m in pyramid_cross:
        pi = m["kinds"].index("pyramid")
        wins_first, wins_last = [], []
        losses_first, losses_last = [], []
        for g in m["games"]:
            d = get_diag_for_policy(g, pi)
            if d is None:
                continue
            per_tick = d.get("value", {}).get("exp_placement", {}).get("per_tick") or []
            if not per_tick:
                continue
            n = len(per_tick)
            seg = max(1, n // 10)
            first = statistics.mean(per_tick[:seg])
            last = statistics.mean(per_tick[-seg:])
            wp = winner_policy_index(g)
            if wp is None:
                continue
            if wp == pi:
                wins_first.append(first)
                wins_last.append(last)
            else:
                losses_first.append(first)
                losses_last.append(last)

        def fmt(xs: list[float]) -> str:
            if not xs:
                return "  n/a  "
            return f"{statistics.mean(xs):5.2f}"

        print()
        print(f"  {m['label']}")
        print(f"    Pyramid wins ({len(wins_first):>2}):  "
              f"start {fmt(wins_first)}  ->  end {fmt(wins_last)}")
        print(f"    Pyramid loss ({len(losses_first):>2}):  "
              f"start {fmt(losses_first)}  ->  end {fmt(losses_last)}")
        if losses_first and losses_last:
            drift = statistics.mean(losses_last) - statistics.mean(losses_first)
            if drift > 1.0:
                interp = "value head adjusted toward defeat — partial calibration"
            elif drift > 0.3:
                interp = "value head shifted slightly — weak calibration"
            else:
                interp = "value head stayed confident even in losses — memorization signature"
            print(f"    -> drift in losses: {drift:+.2f}  ({interp})")


def temp_summary(cross: list[dict]) -> None:
    pyramid_vs_direct = [m for m in cross
                         if set(m["kinds"]) == {"pyramid", "direct"}]
    if len(pyramid_vs_direct) < 2:
        return
    print_header("Pyramid win rate vs temperature")
    print()
    print(f"  {'temp':<6}  {'N':>3}  {'pyramid wins':>13}  {'rate':>7}  {'95% CI':>15}")
    print("-" * 56)
    for m in sorted(pyramid_vs_direct, key=lambda x: float(x["temp"])):
        pi = m["kinds"].index("pyramid")
        n_decided = sum(1 for g in m["games"]
                        if winner_policy_index(g) is not None)
        pw = sum(1 for g in m["games"] if winner_policy_index(g) == pi)
        rate = pw / n_decided if n_decided else float("nan")
        lo, hi = wilson_ci(pw, n_decided) if n_decided else (0.0, 0.0)
        ci = f"{lo*100:4.1f}%-{hi*100:4.1f}%"
        print(f"  {m['temp']:<6}  {n_decided:>3}  {pw:>13}  {rate*100:>6.1f}%  {ci:>15}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", nargs="?", default=None,
                    help="Sweep dir (defaults to latest data/eval-runs/sweep-*)")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else find_latest_sweep()
    if not sweep_dir.exists():
        raise SystemExit(f"Sweep dir not found: {sweep_dir}")
    run_dirs_file = sweep_dir / "run_dirs.txt"
    if not run_dirs_file.exists():
        raise SystemExit(f"Missing {run_dirs_file} — was the sweep interrupted?")

    run_dirs = [Path(line.strip()) for line in run_dirs_file.read_text().splitlines()
                if line.strip()]
    matchups = [load_matchup(d) for d in run_dirs if (d / "results.jsonl").exists()]
    if not matchups:
        raise SystemExit("No matchups with results.jsonl found in sweep")

    cross = [m for m in matchups if not m["is_self"]]
    selfs = [m for m in matchups if m["is_self"]]

    print(f"Sweep:    {sweep_dir}")
    print(f"Matchups: {len(matchups)}  (cross-kind: {len(cross)}, self-vs-self: {len(selfs)})")
    print(f"Total games: {sum(len(m['games']) for m in matchups)}")

    analyze_winrates(cross)
    analyze_self_vs_self(selfs)
    temp_summary(cross)
    analyze_diagnostics(matchups)
    memorization_check(cross)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
