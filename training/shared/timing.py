"""Process-local named timer for perf spikes.

A `Timer` accumulates wall-clock time under string labels — a lightweight,
opt-in alternative to a full profiler for when you want a handful of named
spans ("obs assembly vs the BFS") rather than a per-call trace of everything.
Three front-ends record into the same `name -> (total_ns, calls)` table:

  - `@timer.timed(name)`         — decorate a whole function (the common seam)
  - `with timer.section(name):`  — time a block
  - `timer.start(name)` / `timer.stop(name)` — a manual pair, for spans that
    don't nest cleanly in a `with` (e.g. start and stop in different functions)

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
import functools
import time
import warnings


# Returned by `section` when timing is off, so the disabled path allocates
# nothing. `nullcontext` is re-enterable, so one shared instance is safe even
# when sections nest.
_NULL = nullcontext()


class Timer:
    """Accumulates elapsed ns under string labels. See the module docstring."""

    def __init__(self) -> None:
        self.enabled = False
        self._tot: dict[str, int] = defaultdict(int)   # name -> total ns
        self._cnt: dict[str, int] = defaultdict(int)   # name -> recorded spans
        self._open: dict[str, int] = {}                # name -> start ns (manual)

    def reset(self) -> None:
        """Clear all tallies + any dangling manual timers. Each worker calls
        this at the top of its `__iter__` walk so the flushed file holds that
        epoch's work alone."""
        self._tot.clear()
        self._cnt.clear()
        self._open.clear()

    def snapshot(self) -> dict[str, tuple[int, int]]:
        """`name -> (total_ns, calls)` — a plain dict safe to json-dump / pickle.
        Warns if a manual timer is still open: an unbalanced `start` without a
        `stop` is usually a buggy seam, and its time goes silently unrecorded."""
        if self._open:
            warnings.warn(f"snapshot with timers still open: {list(self._open)}")
        return {n: (self._tot[n], self._cnt[n]) for n in self._tot}

    def _record(self, name: str, ns: int) -> None:
        self._tot[name] += ns
        self._cnt[name] += 1

    def section(self, name: str):
        """Time a `with` block. No-op context when disabled."""
        return self._section(name) if self.enabled else _NULL

    @contextmanager
    def _section(self, name: str):
        t = time.perf_counter_ns()
        try:
            yield
        finally:
            self._record(name, time.perf_counter_ns() - t)

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


# Process-global singleton the seams import. Each process (main + every forked
# DataLoader worker) gets its own instance; `shared.timing_run` flips it on and
# harvests it per-process.
timer = Timer()
