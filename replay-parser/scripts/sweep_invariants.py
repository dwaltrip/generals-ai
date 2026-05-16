"""Sweep the corpus and report sim-invariant violations.

Runs `check_invariants` over each parsed replay. Cross-checks per-timestep
snapshots against the same-game event lists. Expected to produce zero hits;
any hit is actionable.

Usage (from replay-parser/):
    uv run python scripts/sweep_invariants.py [--per-bucket N]
"""
import argparse
from collections import Counter
import sqlite3
import sys

import sim_core

from replay_parser._collector.config import DB_PATH
from replay_parser._collector.wire import decode as decode_blob
from replay_parser._shared import is_vanilla_ffa
from replay_parser.decode import decode_wire_array
from replay_parser.errors import ArmyOverflowError
from replay_parser.invariants import check_invariants

from _sweep_common import bucket_and_sample, fetch_candidates, log

PROGRESS_EVERY = 500
RANDOM_SEED = 42


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=25,
                    help="random sample size per weekly bucket")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        candidates = fetch_candidates(conn, min_version=15)
        log(f"  {len(candidates):,} candidates")
        bucket_pool, sampled = bucket_and_sample(
            candidates,
            per_bucket=args.per_bucket,
            seed=RANDOM_SEED,
        )
        log(f"  sampled {len(sampled):,} replays across {len(bucket_pool)} weekly buckets")
        candidates = sampled

        counts: Counter[str] = Counter()
        first_hits: dict[str, str] = {}
        replays_with_any_violation = 0
        parse_errors = 0
        overflow_errors = 0
        nonvanilla = 0

        for i, (rid, _started, _version) in enumerate(candidates):
            blob = conn.execute(
                "SELECT wire_data FROM replays WHERE id = ?", (rid,)
            ).fetchone()[0]
            try:
                wire = decode_blob(blob)
            except Exception as e:
                parse_errors += 1
                counts[f"decode_err:{type(e).__name__}"] += 1
                first_hits.setdefault(
                    f"decode_err:{type(e).__name__}", f"{rid}: {e}"
                )
                continue
            if not is_vanilla_ffa(wire):
                nonvanilla += 1
                continue
            try:
                replay = decode_wire_array(wire)
                state = sim_core.simulate(replay)
            except ArmyOverflowError:
                overflow_errors += 1
                continue
            except Exception as e:
                parse_errors += 1
                counts[f"parse_err:{type(e).__name__}"] += 1
                first_hits.setdefault(
                    f"parse_err:{type(e).__name__}", f"{rid}: {e}"
                )
                continue

            violations = check_invariants(state, replay)
            if violations:
                replays_with_any_violation += 1
            for vio in violations:
                counts[vio.kind] += 1
                first_hits.setdefault(
                    vio.kind, f"{rid} t={vio.t} {vio.detail}".rstrip()
                )

            if (i + 1) % PROGRESS_EVERY == 0:
                log(f"  {i + 1}/{len(candidates)}  "
                    f"violations so far: {sum(counts.values())} across "
                    f"{replays_with_any_violation} replays")
    finally:
        conn.close()

    log("")
    log(f"Processed: {len(candidates):,}")
    log(f"Non-vanilla (skipped): {nonvanilla:,}")
    log(f"Replays with any violation: {replays_with_any_violation:,}")
    log(f"Overflow (skipped): {overflow_errors:,}")
    log(f"Parse errors: {parse_errors:,}")
    log("")
    if not counts:
        log("  no violations")
    else:
        for kind, n in counts.most_common():
            log(f"  {n:6d}  {kind}")
            log(f"          first: {first_hits.get(kind, '-')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
