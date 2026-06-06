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
from dataclasses import dataclass
import json
from pathlib import Path

from shared.timing import timer


Tally = dict[str, tuple[int, int]]  # name -> (total_ns, calls)


@dataclass
class FileSink:
    """Writes one worker's per-epoch `timer.snapshot()` to its own JSON file.
    Per-worker files sidestep cross-process accumulation: each worker writes
    independently, the driver sums afterward."""

    dir: Path

    def flush(self, epoch: int, wid: int, snap: Tally) -> None:
        (self.dir / f"worker_{epoch}_{wid}.json").write_text(json.dumps(snap))


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
        (prof_dir / "main.json").write_text(json.dumps(main_snap))

    producer = _merge(prof_dir.glob("worker_*.json"))  # explicit glob — never summary.json
    consumer = _merge([prof_dir / "main.json"]) if main_snap else {}

    # Producer seams are all once-per-sample, so the largest call count is the
    # sample count — the denominator that turns summed worker ns into µs/sample.
    n_samples = max((calls for _, calls in producer.values()), default=0)
    _print_producer(producer, n_samples)
    _print_consumer(consumer)

    out = {"producer": producer, "consumer": consumer, "n_samples": n_samples}
    (prof_dir / "summary.json").write_text(json.dumps(out, indent=2))
    return out


def _merge(paths: Iterable[Path]) -> Tally:
    """Sum `{name: [ns, calls]}` files into one tally."""
    tot: dict[str, int] = {}
    cnt: dict[str, int] = {}
    for p in paths:
        for name, (ns, c) in json.loads(p.read_text()).items():
            tot[name] = tot.get(name, 0) + ns
            cnt[name] = cnt.get(name, 0) + c
    return {name: (tot[name], cnt[name]) for name in tot}


def _print_producer(data: Tally, n_samples: int) -> None:
    """Per-region µs/sample (Σns / n_samples) — parallelism-invariant, the
    number to compare across n and before/after a fix. `%share` is within-table
    (regions are disjoint and serial within a worker); the TOTAL is the
    per-sample obs-build cost on one core."""
    print("\nproducer (per sample, summed across workers):")
    if not data:
        print("  (no regions recorded)")
        return
    total_ns = sum(ns for ns, _ in data.values())
    print(f"  {'region':<16} {'us/sample':>10} {'calls':>9} {'share':>7}")
    for name, (ns, calls) in sorted(data.items(), key=lambda kv: -kv[1][0]):
        us = ns / n_samples / 1e3 if n_samples else 0.0
        share = ns / total_ns * 100 if total_ns else 0.0
        print(f"  {name:<16} {us:>10.2f} {calls:>9,} {share:>6.1f}%")
    total_us = total_ns / n_samples / 1e3 if n_samples else 0.0
    print(f"  {'TOTAL':<16} {total_us:>10.2f} {'':>9} {100.0:>6.1f}%")


def _print_consumer(data: Tally) -> None:
    """Main-process regions on their own (serial) denominator — wall-clock per
    batch. `fetch_wait`'s total is the starvation signal: time the GPU sat idle
    waiting on the producer."""
    print("\nconsumer (main process):")
    if not data:
        print("  (no regions recorded)")
        return
    print(f"  {'region':<16} {'total_ms':>10} {'calls':>9} {'mean_ms':>9}")
    for name, (ns, calls) in sorted(data.items(), key=lambda kv: -kv[1][0]):
        mean_ms = ns / calls / 1e6 if calls else 0.0
        print(f"  {name:<16} {ns / 1e6:>10.2f} {calls:>9,} {mean_ms:>9.3f}")
