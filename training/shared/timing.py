"""Process-local named timer for perf spikes.

A `Timer` accumulates wall-clock time under string labels — a lightweight,
opt-in alternative to a full profiler for when you want a handful of named
spans ("obs assembly vs the BFS") rather than a per-call trace of everything.
Three front-ends record into the same per-span accumulator:

  - `@timer.timed(name)`         — decorate a whole function (the common seam)
  - `with timer.section(name):`  — time a block
  - `timer.start(name)` / `timer.stop(name)` — a manual pair, for spans that
    don't nest cleanly in a `with` (e.g. start and stop in different functions)

Each span tracks total/count (for means), min/max (for extremes), and a
sparse log2 histogram (for approximate percentiles). The histogram uses
power-of-2 buckets indexed by `(ns // 10).bit_length()`, flooring at 10 ns
and capping at bucket 39 (~5.5 TiB ns ≈ unreachable). Bucket b covers
the range `[10 * 2^(b-1), 10 * 2^b)` for b >= 1; bucket 0 holds 0–9 ns.

Off by default and switched on per-process — `shared.timing_run` owns
enablement, the cross-worker file fan-out, and reporting. When disabled every
front-end collapses to one branch (the decorator calls straight through, the
manual pair returns immediately, `section` hands back a shared `nullcontext()`),
so seams can sit permanently in hot code at negligible cost.

Deliberately knows nothing about DataLoader workers, files, or the run: a pure
in-process accumulator. Producer seams run inside forked DataLoader workers, so
each process has its own `timer`; getting those per-worker tallies out and
merging them is `shared.timing_run`'s job, not this module's.

Not thread-safe — the accumulators are plain dicts. Our seams live on the
single-threaded DataLoader-worker fetch loop, so this holds; don't drop a
`section` into a multi-threaded path.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import functools
import math
import time
import warnings


# Returned by `section` when timing is off, so the disabled path allocates
# nothing. `nullcontext` is re-enterable, so one shared instance is safe even
# when sections nest.
_NULL = nullcontext()

_HISTO_MAX_BUCKET = 39
_HISTO_FLOOR_NS = 10


def histo_bucket(ns: int) -> int:
    """Map a duration in nanoseconds to a log2 histogram bucket index.

    Floors at 10 ns (anything below goes to bucket 0), then uses
    `bit_length()` as a fast integer log2. Clamped to [0, 39]."""
    if ns < _HISTO_FLOOR_NS:
        return 0
    return min((ns // _HISTO_FLOOR_NS).bit_length(), _HISTO_MAX_BUCKET)


def histo_bucket_upper_ns(bucket: int) -> int:
    """Upper bound (exclusive) of a histogram bucket, in nanoseconds."""
    if bucket <= 0:
        return _HISTO_FLOOR_NS
    return _HISTO_FLOOR_NS * (1 << bucket)


def histo_bucket_midpoint_ns(bucket: int) -> float:
    """Geometric midpoint of a histogram bucket, in nanoseconds."""
    if bucket <= 0:
        return _HISTO_FLOOR_NS / 2.0
    lo = _HISTO_FLOOR_NS * (1 << (bucket - 1))
    hi = _HISTO_FLOOR_NS * (1 << bucket)
    return math.sqrt(lo * hi)


@dataclass
class SpanStats:
    """Per-span accumulated statistics. Returned by `Timer.snapshot()`,
    serialized to JSON by `FileSink`, merged across workers by `timing_run`."""

    total_ns: int
    calls: int
    grouped: bool
    min_ns: int | None = None
    max_ns: int | None = None
    histo: dict[int, int] = field(default_factory=dict)


class Timer:
    """Accumulates elapsed ns under string labels. See the module docstring."""

    def __init__(self) -> None:
        self.enabled = False
        self._tot: dict[str, int] = defaultdict(int)   # name -> total ns
        self._cnt: dict[str, int] = defaultdict(int)   # name -> recorded spans
        self._min: dict[str, int] = {}                  # name -> min ns
        self._max: dict[str, int] = {}                  # name -> max ns
        self._histo: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._open: dict[str, int] = {}                 # name -> start ns (manual)
        self._ungrouped: set[str] = set()               # names recorded grouped=False

    def reset(self) -> None:
        """Clear all tallies + any dangling manual timers. Each worker calls
        this at the top of its `__iter__` walk so the flushed file holds that
        epoch's work alone."""
        self._tot.clear()
        self._cnt.clear()
        self._min.clear()
        self._max.clear()
        self._histo.clear()
        self._open.clear()
        self._ungrouped.clear()

    def snapshot(self) -> dict[str, SpanStats]:
        """`name -> SpanStats` — a plain dict safe to serialize. Warns if a
        manual timer is still open: an unbalanced `start` without a `stop` is
        usually a buggy seam, and its time goes silently unrecorded."""
        if self._open:
            warnings.warn(f"snapshot with timers still open: {list(self._open)}")
        return {
            n: SpanStats(
                total_ns=self._tot[n],
                calls=self._cnt[n],
                grouped=n not in self._ungrouped,
                min_ns=self._min.get(n),
                max_ns=self._max.get(n),
                histo=dict(self._histo.get(n, {})),
            )
            for n in self._tot
        }

    def _record(self, name: str, ns: int, grouped: bool = True) -> None:
        self._tot[name] += ns
        self._cnt[name] += 1
        prev_min = self._min.get(name)
        if prev_min is None or ns < prev_min:
            self._min[name] = ns
        prev_max = self._max.get(name)
        if prev_max is None or ns > prev_max:
            self._max[name] = ns
        self._histo[name][histo_bucket(ns)] += 1
        if not grouped:
            self._ungrouped.add(name)

    def section(self, name: str, grouped: bool = True):
        """Time a `with` block. No-op context when disabled.

        `grouped=False` marks the span as overlapping/reference — still recorded,
        but the reporter keeps it out of the grouped TOTAL and %share. Use it for
        a parent span wrapping other seams (e.g. `encode_frame` over its
        children), where summing both would double-count."""
        return self._section(name, grouped) if self.enabled else _NULL

    @contextmanager
    def _section(self, name: str, grouped: bool = True):
        t = time.perf_counter_ns()
        try:
            yield
        finally:
            self._record(name, time.perf_counter_ns() - t, grouped)

    def timed(self, name: str):
        """Decorator form of `section` for whole functions. Reads `enabled` at
        call time, not decoration time — a seam decorated at import can be
        switched on later inside a worker."""
        def deco(fn):
            @functools.wraps(fn)
            def wrap(*args, **kwargs):
                if not self.enabled:
                    return fn(*args, **kwargs)
                with self._section(name):
                    return fn(*args, **kwargs)
            return wrap
        return deco

    def start(self, name: str) -> None:
        """Open a manual timer; pair with `stop(name)`. For spans a `with`
        can't express (e.g. opened in one function, closed in another)."""
        if not self.enabled:
            return
        if name in self._open:
            warnings.warn(f"start({name!r}): already running")
        self._open[name] = time.perf_counter_ns()

    def stop(self, name: str) -> None:
        """Close a manual timer opened by `start(name)`; warns (and no-ops) when
        nothing is open under that name."""
        if not self.enabled:
            return
        t = self._open.pop(name, None)
        if t is None:
            warnings.warn(f"stop({name!r}): no timer running")
            return
        self._record(name, time.perf_counter_ns() - t)

    def add(self, name: str, ns: int) -> None:
        """Record an already-measured duration as a single span under `name` —
        for timings the wall-clock front-ends can't capture, e.g. an async GPU
        copy clocked with CUDA events."""
        if not self.enabled:
            return
        self._record(name, ns)


# Process-global singleton the seams import. Each process (main + every forked
# DataLoader worker) gets its own instance; `shared.timing_run` flips it on and
# harvests it per-process.
timer = Timer()
