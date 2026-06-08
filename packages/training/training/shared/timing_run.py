"""Run-lifecycle glue for the named-span `timer`: per-worker file sink, merge,
and report.

`shared.timing` is the pure in-process accumulator; this module is the part that
knows about the run. It enables the timer, fans per-worker tallies out to JSON
files — the only way to recover producer timings from forked DataLoader workers,
which each hold their own `timer` — and merges them back into a report.

Lifecycle, driven by the entry script's `--profile` flag:

    begin(prof_dir)           # main: clear stale files, register the sink, enable
    ... training run ...       # workers flush per-epoch files via the sink;
                               #   main-process consumer seams accumulate in `timer`
    end_and_report(prof_dir)  # main: merge worker files + main tally, print, dump

`active_sink()` is how `build_dataloader` reaches the registered sink without
threading it through the train-config call chain; it rides to each worker on the
pickled dataset instance, where its presence is the per-process enable signal.

The producer (worker) and consumer (main) tables are reported with *separate*
denominators: worker time is summed across N parallel workers (overlapping
wall-clock) while main time is serial, so folding both into one %share would
inflate the producer ~N-fold.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from training.shared.timing import SpanStats, timer


Tally = dict[str, SpanStats]


def _span_to_json(s: SpanStats) -> dict:
    """Serialize a SpanStats to a JSON-safe dict. Histogram keys are ints in
    Python but become strings in JSON; the read path coerces them back."""
    return asdict(s)


def _span_from_json(d: dict | list) -> SpanStats:
    """Deserialize a span from JSON. Handles both the current dict format and
    the legacy [ns, calls, grouped?] list format from older summary.json files."""
    if isinstance(d, list):
        return SpanStats(
            total_ns=d[0], calls=d[1],
            grouped=d[2] if len(d) > 2 else True,
        )
    histo = d.get("histo") or {}
    return SpanStats(
        total_ns=d["total_ns"], calls=d["calls"],
        grouped=d.get("grouped", True),
        min_ns=d.get("min_ns"), max_ns=d.get("max_ns"),
        histo={int(k): v for k, v in histo.items()},
    )


def _tally_to_json(tally: Tally) -> dict:
    return {name: _span_to_json(s) for name, s in tally.items()}


def _tally_from_json(raw: dict) -> Tally:
    return {name: _span_from_json(v) for name, v in raw.items()}


@dataclass
class FileSink:
    """Writes one worker's per-epoch `timer.snapshot()` to its own JSON file.
    Per-worker files sidestep cross-process accumulation: each worker writes
    independently, the driver sums afterward."""

    dir: Path

    def flush(self, epoch: int, wid: int, snap: dict[str, SpanStats]) -> None:
        data = {name: _span_to_json(s) for name, s in snap.items()}
        (self.dir / f"worker_{epoch}_{wid}.json").write_text(json.dumps(data))


_active: FileSink | None = None


def begin(prof_dir: Path) -> None:
    """Open a profiling window (main process). Clears stale tallies from the dir
    (runs reuse the run_dir on resume), registers the sink for `active_sink`,
    and enables the main-process `timer` for consumer seams."""
    global _active
    prof_dir.mkdir(parents=True, exist_ok=True)
    for p in prof_dir.glob("*.json"):
        p.unlink()
    _active = FileSink(prof_dir)
    timer.enabled = True


def active_sink() -> FileSink | None:
    """The sink registered by `begin`, or None when not profiling. Read by
    `build_dataloader` to hand the sink to the dataset (and thence each worker)."""
    return _active


def end_and_report(prof_dir: Path) -> dict:
    """Close the window (main process): flush the main-process consumer tally,
    merge it and the per-worker producer files into two tables, print them, and
    persist `summary.json`."""
    main_snap = timer.snapshot()
    if main_snap:
        data = {name: _span_to_json(s) for name, s in main_snap.items()}
        (prof_dir / "main.json").write_text(json.dumps(data))

    producer = _merge(prof_dir.glob("worker_*.json"))  # explicit glob — never summary.json
    consumer = _merge([prof_dir / "main.json"]) if main_snap else {}

    # Per-sample seams are the most frequent, so the largest call count is the
    # sample count — the denominator that turns summed worker ns into µs/sample.
    # Ungrouped spans are excluded: a parent like `encode_frame` shares the
    # per-sample count, but a coarser one (per-batch/game) must not anchor it.
    n_samples = max(
        (s.calls for s in producer.values() if s.grouped),
        default=0,
    )
    _print_producer(producer, n_samples)
    _print_consumer(consumer)

    out = {
        "producer": _tally_to_json(producer),
        "consumer": _tally_to_json(consumer),
        "n_samples": n_samples,
    }
    (prof_dir / "summary.json").write_text(json.dumps(out, indent=2))
    return out


def _merge(paths: Iterable[Path]) -> Tally:
    """Merge per-worker span files into one tally: sum totals/counts/histograms,
    take min of mins, max of maxes."""
    merged: dict[str, SpanStats] = {}
    for p in paths:
        for name, s in _tally_from_json(json.loads(p.read_text())).items():
            if name not in merged:
                merged[name] = SpanStats(
                    total_ns=s.total_ns, calls=s.calls, grouped=s.grouped,
                    min_ns=s.min_ns, max_ns=s.max_ns,
                    histo=dict(s.histo),
                )
            else:
                m = merged[name]
                m.total_ns += s.total_ns
                m.calls += s.calls
                if s.min_ns is not None:
                    m.min_ns = s.min_ns if m.min_ns is None else min(m.min_ns, s.min_ns)
                if s.max_ns is not None:
                    m.max_ns = s.max_ns if m.max_ns is None else max(m.max_ns, s.max_ns)
                for bucket, count in s.histo.items():
                    m.histo[bucket] = m.histo.get(bucket, 0) + count
    return merged


def _print_producer(data: Tally, n_samples: int) -> None:
    """Per-region µs/sample (Σns / n_samples) — parallelism-invariant, the
    number to compare across n and before/after a fix. `%share` is within-table
    (regions are disjoint and serial within a worker); the TOTAL is the
    per-sample obs-build cost on one core. Ungrouped (`grouped=False`) spans
    overlap the grouped ones, so they print below the TOTAL as reference rows."""
    print("\nproducer (per sample, summed across workers):")
    if not data:
        print("  (no regions recorded)")
        return
    grouped = {n: s for n, s in data.items() if s.grouped}
    ungrouped = {n: s for n, s in data.items() if not s.grouped}
    total_ns = sum(s.total_ns for s in grouped.values())
    print(f"  {'region':<16} {'us/sample':>10} {'calls':>9} {'share':>7}")
    for name, s in sorted(grouped.items(), key=lambda kv: -kv[1].total_ns):
        us = s.total_ns / n_samples / 1e3 if n_samples else 0.0
        share = s.total_ns / total_ns * 100 if total_ns else 0.0
        print(f"  {name:<16} {us:>10.2f} {s.calls:>9,} {share:>6.1f}%")
    total_us = total_ns / n_samples / 1e3 if n_samples else 0.0
    print(f"  {'TOTAL':<16} {total_us:>10.2f} {'':>9} {100.0:>6.1f}%")
    for name, s in sorted(ungrouped.items(), key=lambda kv: -kv[1].total_ns):
        us = s.total_ns / n_samples / 1e3 if n_samples else 0.0
        print(f"  {name:<16} {us:>10.2f} {s.calls:>9,} {'(ref)':>7}")


def _print_consumer(data: Tally) -> None:
    """Main-process regions on their own (serial) denominator — wall-clock per
    batch. `fetch_wait`'s total is the starvation signal: time the GPU sat idle
    waiting on the producer."""
    print("\nconsumer (main process):")
    if not data:
        print("  (no regions recorded)")
        return
    print(f"  {'region':<16} {'total_ms':>10} {'calls':>9} {'mean_ms':>9}")
    for name, s in sorted(data.items(), key=lambda kv: -kv[1].total_ns):
        mean_ms = s.total_ns / s.calls / 1e6 if s.calls else 0.0
        print(f"  {name:<16} {s.total_ns / 1e6:>10.2f} {s.calls:>9,} {mean_ms:>9.3f}")
