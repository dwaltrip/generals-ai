"""Sweep the corpus and report Rust-vs-Python sim parity rates by week.

Runs `sim_core.simulate(replay)` and `parse_replay(blob)` on the same
v15+ FFA replays, diffs every snapshot + event list + damage matrix,
and writes a weekly markdown report to replay-parser/tmp/.

Parity is a per-replay property — either every snapshot/event/matrix
matches, or one doesn't and we record which. The v30.9.2 lbSort cutoff
is a ranking concern, not a sim concern, so we include the ambiguity
window in the sample (week buckets give date-range coverage).

Usage (from replay-parser/):
    uv run python scripts/parity_sweep.py [--per-bucket 150] [--seed 42]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import numpy as np
import sim_core
from tabulate import tabulate

from replay_parser._collector.config import DB_PATH
from replay_parser._collector.wire import decode as decode_blob
from replay_parser._shared import is_vanilla_ffa
from replay_parser.errors import ArmyOverflowError
from replay_parser.parser import parse_replay as parse_python

from _sweep_common import (
    ReplayInfo,
    bucket_and_sample,
    fetch_blobs,
    fetch_candidates,
    log,
    week_start,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "tmp"
PROGRESS_EVERY = 100
DEFAULT_PER_BUCKET = 150
DEFAULT_SEED = 42
MAX_ERROR_SAMPLES = 20
MAX_FAIL_IDS_PER_BUCKET = 3


@dataclass
class Bucket:
    total: int = 0
    pass_: int = 0
    fail: int = 0
    overflow: int = 0
    nonvanilla: int = 0
    parse_err: int = 0
    versions: set[int] = field(default_factory=set)
    fail_ids: list[str] = field(default_factory=list)


@dataclass
class SweepData:
    total_candidates: int
    sampled_replays: list[ReplayInfo]
    blobs_by_id: dict[str, bytes]


# ============================================================================
# Parity check — lifted from tmp/parity_b2.py. Will be deleted post-swap
# when the Python sim is removed.
# ============================================================================

def _diff_snapshots(rs_list, py_list, name: str) -> str | None:
    if len(rs_list) != len(py_list):
        return f"{name}: snapshot count differs (rs={len(rs_list)} py={len(py_list)})"
    for t, (a, b) in enumerate(zip(rs_list, py_list)):
        if not np.array_equal(a, b):
            diffs = np.flatnonzero(a != b)
            return (
                f"{name}: differ at t={t}, {len(diffs)} cells "
                f"(first idx={diffs[0]}, rs={a[diffs[0]]}, py={b[diffs[0]]})"
            )
    return None


def _diff_events(rs_events, py_events, name: str, fields: tuple[str, ...]) -> str | None:
    if len(rs_events) != len(py_events):
        return f"{name}: count differs (rs={len(rs_events)} py={len(py_events)})"
    for i, (r, p) in enumerate(zip(rs_events, py_events)):
        for f in fields:
            if getattr(r, f) != getattr(p, f):
                return f"{name}[{i}].{f} differs (rs={getattr(r, f)} py={getattr(p, f)})"
    return None


def parity_check(blob: bytes) -> str | None:
    """Returns None on full parity, or a short divergence summary. Raises
    ArmyOverflowError if both sims agree on overflow (the caller bookkeeps
    that case separately)."""
    py_state, replay = parse_python(blob)
    try:
        rs_state = sim_core.simulate(replay)
    except ArmyOverflowError:
        # Both sims agreed on overflow before/at this point — that's parity.
        return None

    if rs_state.timestep != py_state.timestep:
        return f"timestep diverged: rs={rs_state.timestep} py={py_state.timestep}"
    if rs_state.alive != py_state.alive:
        return f"alive diverged: rs={rs_state.alive} py={py_state.alive}"
    if rs_state.has_kill != py_state.has_kill:
        return f"has_kill diverged: rs={rs_state.has_kill} py={py_state.has_kill}"

    if (msg := _diff_snapshots(rs_state.snapshots_ownership, py_state.snapshots.ownership, "ownership")):
        return msg
    if (msg := _diff_snapshots(rs_state.snapshots_armies, py_state.snapshots.armies, "armies")):
        return msg
    py_cm = [a.astype(np.uint8) for a in py_state.snapshots.cities_mask]
    if (msg := _diff_snapshots(rs_state.snapshots_cities_mask, py_cm, "cities_mask")):
        return msg

    if (msg := _diff_events(rs_state.death_events, py_state.death_events, "death_events", ("timestep", "player"))):
        return msg
    if (msg := _diff_events(rs_state.capture_events, py_state.capture_events, "capture_events", ("timestep", "captor", "captured"))):
        return msg
    if (msg := _diff_events(rs_state.neutralize_events, py_state.neutralize_events, "neutralize_events", ("timestep", "player"))):
        return msg

    for label, rs_attr, py_attr in [
        ("damage_sym_all", rs_state.damage_sym_all, py_state.damage_sym_all),
        ("damage_sym_pre", rs_state.damage_sym_pre, py_state.damage_sym_pre),
        ("damage_off_all", rs_state.damage_off_all, py_state.damage_off_all),
        ("damage_off_pre", rs_state.damage_off_pre, py_state.damage_off_pre),
    ]:
        if not np.array_equal(rs_attr, py_attr):
            return f"{label} differs"

    return None


# ============================================================================
# Sweep + report
# ============================================================================

def get_sweep_data(*, per_bucket: int, seed: int) -> SweepData:
    conn = sqlite3.connect(DB_PATH)
    try:
        log("Fetching candidate metadata...")
        candidates = fetch_candidates(conn, min_version=15)
        total_candidates = len(candidates)
        log(f"  {total_candidates:,} candidate replays")

        bucket_pool, sampled = bucket_and_sample(
            candidates, per_bucket=per_bucket, seed=seed
        )
        log(f"  sampled {len(sampled):,} replays across {len(bucket_pool)} weeks")

        sampled_ids = [r[0] for r in sampled]
        log("Fetching blobs for sampled replays...")
        blobs = fetch_blobs(conn, sampled_ids)
        log(f"  fetched {len(blobs):,} blobs")
    finally:
        conn.close()

    return SweepData(
        total_candidates=total_candidates,
        sampled_replays=sampled,
        blobs_by_id=blobs,
    )


def write_report(
    buckets: dict[str, Bucket],
    *,
    sampled_count: int,
    total_candidates: int,
    per_bucket: int,
    seed: int,
    failure_samples: list[tuple[str, str]],
    parse_error_samples: list[tuple[str, str, str]],
    out_dir: Path,
) -> Path:
    table_rows = []
    for wk in sorted(buckets):
        b = buckets[wk]
        denom = b.pass_ + b.fail
        pct = f"{100 * b.pass_ / denom:.1f}%" if denom else "-"
        versions = "{" + ",".join(str(v) for v in sorted(b.versions)) + "}"
        sample = " ".join(b.fail_ids)
        table_rows.append([
            wk, b.total, b.pass_, b.fail, b.overflow,
            b.nonvanilla, b.parse_err, pct, versions, sample,
        ])

    totals_row = [
        "TOTAL",
        sum(b.total for b in buckets.values()),
        sum(b.pass_ for b in buckets.values()),
        sum(b.fail for b in buckets.values()),
        sum(b.overflow for b in buckets.values()),
        sum(b.nonvanilla for b in buckets.values()),
        sum(b.parse_err for b in buckets.values()),
        "", "", "",
    ]
    denom = totals_row[2] + totals_row[3]
    totals_row[7] = f"{100 * totals_row[2] / denom:.1f}%" if denom else "-"
    table_rows.append(totals_row)

    table = tabulate(
        table_rows,
        headers=[
            "week (mon)", "total", "pass", "fail", "overflow",
            "non-vanilla", "parse-err", "%pass", "versions", "sample fail ids",
        ],
        tablefmt="github",
    )

    now = datetime.now(tz=timezone.utc)
    lines = [
        "# Sweep: Rust-vs-Python sim parity rates",
        "",
        f"Generated: {now.isoformat()}",
        f"Total v15+ FFA candidates: {total_candidates:,}",
        f"Sample target: {per_bucket}/week (random, seed={seed})",
        f"Sampled: {sampled_count:,} across {len(buckets)} buckets",
        "",
        "%pass denominator = pass + fail (excludes overflow, non-vanilla, parse-err).",
        "",
        table,
        "",
    ]
    if failure_samples:
        lines.extend(["", f"## Parity failures (first {len(failure_samples)})", ""])
        for rid, msg in failure_samples:
            lines.append(f"- `{rid}`  {msg}")
        lines.append("")
    if parse_error_samples:
        lines.extend(["", f"## Parse error samples (first {len(parse_error_samples)})", ""])
        for rid, err_type, msg in parse_error_samples:
            lines.append(f"- `{rid}`  {err_type}: {msg}")
        lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"parity_sweep-{now.strftime('%Y%m%d-%H%M')}.md"
    out_path.write_text("\n".join(lines))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=DEFAULT_PER_BUCKET)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    data = get_sweep_data(per_bucket=args.per_bucket, seed=args.seed)
    total = len(data.sampled_replays)

    buckets: dict[str, Bucket] = defaultdict(Bucket)
    failure_samples: list[tuple[str, str]] = []
    parse_error_samples: list[tuple[str, str, str]] = []

    processed = 0
    for replay_id, started, version in data.sampled_replays:
        processed += 1
        if processed % PROGRESS_EVERY == 0:
            log(f"  ... {processed:,}/{total:,}")

        b = buckets[week_start(started)]
        b.total += 1
        b.versions.add(version)

        blob = data.blobs_by_id[replay_id]
        try:
            wire = decode_blob(blob)
        except Exception as e:
            b.parse_err += 1
            if len(parse_error_samples) < MAX_ERROR_SAMPLES:
                parse_error_samples.append((replay_id, type(e).__name__, str(e)))
            continue

        if not is_vanilla_ffa(wire):
            b.nonvanilla += 1
            continue

        try:
            msg = parity_check(blob)
        except ArmyOverflowError:
            b.overflow += 1
            continue
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            # pyo3 PanicException inherits from BaseException, not Exception —
            # catching it as a parity divergence rather than a parse error.
            b.parse_err += 1
            if len(parse_error_samples) < MAX_ERROR_SAMPLES:
                parse_error_samples.append((replay_id, type(e).__name__, str(e)))
            continue

        if msg is None:
            b.pass_ += 1
        else:
            b.fail += 1
            if len(b.fail_ids) < MAX_FAIL_IDS_PER_BUCKET:
                b.fail_ids.append(replay_id)
            if len(failure_samples) < MAX_ERROR_SAMPLES:
                failure_samples.append((replay_id, msg))

    out_path = write_report(
        buckets,
        sampled_count=total,
        total_candidates=data.total_candidates,
        per_bucket=args.per_bucket,
        seed=args.seed,
        failure_samples=failure_samples,
        parse_error_samples=parse_error_samples,
        out_dir=OUT_DIR,
    )

    log(f"Wrote: {out_path}")
    log(f"Total candidates: {data.total_candidates:,}")
    log(f"Sampled: {total:,}")
    log(f"  pass: {sum(b.pass_ for b in buckets.values()):,}")
    log(f"  fail: {sum(b.fail for b in buckets.values()):,}")
    log(f"  overflow: {sum(b.overflow for b in buckets.values()):,}")
    log(f"  non-vanilla: {sum(b.nonvanilla for b in buckets.values()):,}")
    log(f"  parse-err: {sum(b.parse_err for b in buckets.values()):,}")
    log(f"  buckets: {len(buckets)}")


if __name__ == "__main__":
    main()
