"""Sample ~50 random FFA v15+ replays from the collector DB and report the
size distribution of `write_sim_output` + `write_metadata` output.

Usage (from replay-parser/):
    uv run python tmp/sample_output_sizes.py [--n 50] [--seed 7]
"""
import argparse
from pathlib import Path
import sqlite3
import statistics
import sys
import tempfile

import numpy as np

from replay_parser._collector.config import DB_PATH
from replay_parser.metadata import build_metadata
from replay_parser.output import write_metadata, write_sim_output
from replay_parser.parser import parse_replay


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="sample size")
    ap.add_argument(
        "--seed",
        type=int,
        default=7,
        help="sqlite RANDOM() seed (not strictly applied; sqlite RANDOM() is non-seedable)",
    )
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            """
            SELECT id, wire_data, turns, player_count, version
            FROM replays
            WHERE ladder_id = 'ffa'
              AND version >= 15
              AND wire_data IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (args.n,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("No rows returned — check ladder_id / version filter.", file=sys.stderr)
        return 1

    results: list[dict] = []
    print(f"Sampled {len(rows)} replays. Writing to a temp dir.\n")
    print(
        f"{'replay_id':<12} {'P':>2} {'T':>5} {'sim_KB':>7} {'meta_B':>7}",
        f"{'wire_KB':>8} {'sim/T_B':>8}",
    )
    print("-" * 60)
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        for rid, wire, _turns, _pcount, ver in rows:
            try:
                state, replay = parse_replay(wire)
            except Exception as e:
                print(f"{rid:<12}  parse fail: {type(e).__name__}: {e}")
                continue
            sim_path = out_dir / f"{rid}.npz"
            meta_path = out_dir / f"{rid}.meta.npz"
            write_sim_output(state, replay, sim_path)
            P = state.num_players
            meta = build_metadata(
                state, replay,
                perspective_player_ids=list(range(P)),
                placement=list(range(1, P + 1)),
                sim_core_version="size-sample",
            )
            write_metadata(meta, meta_path)

            T = state.snapshots_len
            sim_b = sim_path.stat().st_size
            meta_b = meta_path.stat().st_size
            wire_b = len(wire)
            results.append({
                "id": rid, "T": T, "P": P, "version": ver,
                "sim_b": sim_b, "meta_b": meta_b, "wire_b": wire_b,
            })
            print(
                f"{rid:<12} {P:>2} {T:>5} {sim_b/1024:>7.1f} {meta_b:>7} "
                f"{wire_b/1024:>8.1f} {sim_b/max(T,1):>8.1f}"
            )

    if not results:
        print("\nNo results.")
        return 1

    sim_bs = np.array([r["sim_b"] for r in results])
    meta_bs = np.array([r["meta_b"] for r in results])
    Ts = np.array([r["T"] for r in results])
    wires = np.array([r["wire_b"] for r in results])
    sim_per_t = sim_bs / np.maximum(Ts, 1)

    def summarize(label: str, xs: np.ndarray, unit: str, fmt: str = ".1f"):
        q = np.quantile(xs, [0.0, 0.5, 0.9, 0.95, 1.0])
        print(
            f"  {label:<12} "
            f"min={q[0]:{fmt}} p50={q[1]:{fmt}} p90={q[2]:{fmt}} "
            f"p95={q[3]:{fmt}} max={q[4]:{fmt}} "
            f"mean={xs.mean():{fmt}} stdev={statistics.pstdev(xs.tolist()):{fmt}} {unit}"
        )

    print("\n=== Distribution ===")
    summarize("sim KB",     sim_bs / 1024, "KB")
    summarize("meta B",     meta_bs, "B", ".0f")
    summarize("wire KB",    wires / 1024, "KB")
    summarize("T (snaps)",  Ts, "snaps", ".0f")
    summarize("sim B/snap", sim_per_t, "B/snap")

    n = len(results)
    total_sim_mb = sim_bs.sum() / 1024 / 1024
    total_meta_mb = meta_bs.sum() / 1024 / 1024
    mean_sim_kb = sim_bs.mean() / 1024
    print(
        f"\n=== Totals ({n} games) ===\n"
        f"  sim   total = {total_sim_mb:.2f} MB   per-game mean = {mean_sim_kb:.1f} KB\n"
        f"  meta  total = {total_meta_mb:.2f} MB   per-game mean = {meta_bs.mean():.0f} B\n"
    )

    # Linear extrapolation to ~180k v15+ FFA corpus (per memory).
    corpus_n = 180_000
    proj_sim_gb = mean_sim_kb * corpus_n / 1024 / 1024
    proj_meta_gb = meta_bs.mean() * corpus_n / 1024 / 1024 / 1024
    print(
        f"=== Linear projection to {corpus_n:,} games ===\n"
        f"  sim   ≈ {proj_sim_gb:.1f} GB\n"
        f"  meta  ≈ {proj_meta_gb:.2f} GB\n"
    )

    print("(For reference, 5.16-3 §5 cites the design-doc envelope of 50–65 GB,\n"
          "which was computed *with* a per-timestep cities mask we've dropped.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
