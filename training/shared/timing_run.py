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


def end_and_report(prof_dir: Path) -> dict[str, Tally]:
    """Close the window (main process): flush the main-process consumer tally,
    merge it and the per-worker producer files into two tables, print them, and
    persist `summary.json`."""
    main_snap = timer.snapshot()
    if main_snap:
        (prof_dir / "main.json").write_text(json.dumps(main_snap))

    producer = _merge(prof_dir.glob("worker_*.json"))  # explicit glob — never summary.json
    consumer = _merge([prof_dir / "main.json"]) if main_snap else {}

    _print_table("producer (summed across workers)", producer)
    _print_table("consumer (main process)", consumer)

    out = {"producer": producer, "consumer": consumer}
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


def _print_table(title: str, data: Tally) -> None:
    """Print one region table sorted by total time. `%share` is within-table —
    valid because every region here is serial within its process. Per-sample
    normalization (µs/sample via an anchor region's count) lands with the
    producer seams; until then this shows totals, calls, and mean per call."""
    print(f"\n{title}:")
    if not data:
        print("  (no regions recorded)")
        return
    total_ns = sum(ns for ns, _ in data.values())
    print(f"  {'region':<24} {'total_ms':>10} {'calls':>9} {'mean_us':>9} {'share':>7}")
    for name, (ns, calls) in sorted(data.items(), key=lambda kv: -kv[1][0]):
        mean_us = ns / calls / 1e3 if calls else 0.0
        share = ns / total_ns * 100 if total_ns else 0.0
        print(f"  {name:<24} {ns / 1e6:>10.2f} {calls:>9,} {mean_us:>9.2f} {share:>6.1f}%")
