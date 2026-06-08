"""Unit coverage for the named-span Timer (`shared.timing`).

Tests use fresh `Timer()` instances rather than the module-global `timer`, so
each case is isolated from the process-global state.
"""

from __future__ import annotations

import threading

import pytest

from training.shared.timing import (
    _HISTO_FLOOR_NS,
    _HISTO_MAX_BUCKET,
    LockedTimer,
    SpanStats,
    Timer,
    histo_bucket,
    histo_bucket_midpoint_ns,
    histo_bucket_upper_ns,
    timer,
)


def test_disabled_records_nothing():
    t = Timer()  # enabled is False by default

    with t.section("a"):
        pass
    t.start("b")
    t.stop("b")

    @t.timed("c")
    def f() -> int:
        return 42

    assert f() == 42  # decorated fn still calls through when disabled
    assert t.snapshot() == {}


def test_records_when_enabled():
    t = Timer()
    t.enabled = True

    with t.section("sec"):
        pass
    with t.section("sec"):
        pass
    t.start("man")
    t.stop("man")

    @t.timed("fn")
    def f() -> str:
        return "x"

    assert f() == "x"

    snap = t.snapshot()
    assert set(snap) == {"sec", "man", "fn"}
    assert snap["sec"].calls == 2
    assert snap["man"].calls == 1
    assert snap["fn"].calls == 1
    assert all(s.total_ns >= 0 for s in snap.values())


def test_snapshot_returns_span_stats():
    t = Timer()
    t.enabled = True
    with t.section("a"):
        pass
    snap = t.snapshot()
    s = snap["a"]
    assert isinstance(s, SpanStats)
    assert s.calls == 1
    assert s.grouped is True
    assert s.min_ns is not None
    assert s.max_ns is not None
    assert isinstance(s.histo, dict)
    assert sum(s.histo.values()) == 1


def test_grouped_flag():
    t = Timer()
    t.enabled = True
    with t.section("leaf"):
        pass
    with t.section("parent", grouped=False):  # overlapping/reference span
        pass
    snap = t.snapshot()
    assert snap["leaf"].grouped is True
    assert snap["parent"].grouped is False


def test_min_max_tracking():
    t = Timer()
    t.enabled = True
    t.add("x", 100)
    t.add("x", 500)
    t.add("x", 200)
    snap = t.snapshot()
    assert snap["x"].min_ns == 100
    assert snap["x"].max_ns == 500


def test_histogram_accumulation():
    t = Timer()
    t.enabled = True
    t.add("x", 50)
    t.add("x", 50)
    t.add("x", 5_000_000)
    snap = t.snapshot()
    assert sum(snap["x"].histo.values()) == 3
    # The two 50 ns values should land in the same bucket
    b_small = histo_bucket(50)
    assert snap["x"].histo[b_small] == 2
    # The 5 ms value should land in a different bucket
    b_big = histo_bucket(5_000_000)
    assert b_big > b_small
    assert snap["x"].histo[b_big] == 1


def test_reset_clears():
    t = Timer()
    t.enabled = True
    with t.section("a"):
        pass
    t.reset()
    assert t.snapshot() == {}


def test_stop_without_start_warns():
    t = Timer()
    t.enabled = True
    with pytest.warns(UserWarning, match="no timer running"):
        t.stop("nope")
    assert t.snapshot() == {}


def test_double_start_warns():
    t = Timer()
    t.enabled = True
    t.start("x")
    with pytest.warns(UserWarning, match="already running"):
        t.start("x")
    t.stop("x")


def test_snapshot_warns_on_open_timer():
    t = Timer()
    t.enabled = True
    t.start("dangling")
    with pytest.warns(UserWarning, match="still open"):
        snap = t.snapshot()
    # An open timer hasn't recorded yet, so it's absent from the tally.
    assert "dangling" not in snap


def test_add_records_external_duration():
    t = Timer()
    t.enabled = True
    t.add("h2d", 5000)
    t.add("h2d", 3000)
    s = t.snapshot()["h2d"]
    assert s.total_ns == 8000
    assert s.calls == 2
    assert s.grouped is True
    assert s.min_ns == 3000
    assert s.max_ns == 5000


def test_add_noop_when_disabled():
    t = Timer()  # disabled
    t.add("h2d", 5000)
    assert t.snapshot() == {}


def test_timed_preserves_function_identity():
    t = Timer()

    @t.timed("f")
    def my_func() -> int:
        """A docstring."""
        return 1

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "A docstring."


def test_module_singleton_disabled_by_default():
    # Seams import this global; it must be inert in production runs.
    assert timer.enabled is False


# ---- Histogram bucket helpers ----

def test_histo_bucket_below_floor():
    assert histo_bucket(0) == 0
    assert histo_bucket(5) == 0
    assert histo_bucket(9) == 0


def test_histo_bucket_at_floor():
    # 10 // 10 = 1, bit_length() = 1
    assert histo_bucket(10) == 1
    assert histo_bucket(19) == 1


def test_histo_bucket_progression():
    # 20 // 10 = 2, bit_length() = 2
    assert histo_bucket(20) == 2
    assert histo_bucket(39) == 2
    # 40 // 10 = 4, bit_length() = 3
    assert histo_bucket(40) == 3


def test_histo_bucket_large_values():
    # 1 second = 1e9 ns; 1e9 // 10 = 1e8, bit_length = 27
    b = histo_bucket(1_000_000_000)
    assert b == 27
    # 100 seconds = 1e11 ns
    b100 = histo_bucket(100_000_000_000)
    assert b100 == 34


def test_histo_bucket_clamps_at_max():
    huge = 10**18
    assert histo_bucket(huge) == _HISTO_MAX_BUCKET


def test_histo_bucket_upper_ns():
    assert histo_bucket_upper_ns(0) == _HISTO_FLOOR_NS
    assert histo_bucket_upper_ns(1) == _HISTO_FLOOR_NS * 2
    assert histo_bucket_upper_ns(2) == _HISTO_FLOOR_NS * 4


def test_histo_bucket_midpoint_ns():
    mid = histo_bucket_midpoint_ns(2)
    # bucket 2: range [10*2, 10*4) = [20, 40)
    assert 20 < mid < 40


# ---- LockedTimer (thread-safe recording) ----

def test_locked_timer_no_lost_updates():
    # The whole reason LockedTimer exists: concurrent `add`s must not drop any
    # read-modify-write update. With the lock, the totals are exact.
    t = LockedTimer()
    t.enabled = True
    n_threads, n_adds, dur = 8, 5000, 100

    def worker() -> None:
        for _ in range(n_adds):
            t.add("x", dur)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    s = t.snapshot()["x"]
    assert s.calls == n_threads * n_adds
    assert s.total_ns == n_threads * n_adds * dur


def test_locked_timer_snapshot_no_raise_during_writes():
    # snapshot() iterates the span dicts; without the lock a concurrent add()
    # could trigger "dict changed size during iteration". It must not.
    t = LockedTimer()
    t.enabled = True
    stop = threading.Event()
    errors: list[Exception] = []

    def writer() -> None:
        try:
            i = 0
            while not stop.is_set():
                t.add(f"span_{i % 32}", 50)  # grow the dict over time
                i += 1
        except Exception as e:  # noqa: BLE001 - surfaced via the assert below
            errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for th in threads:
        th.start()
    try:
        for _ in range(500):
            t.snapshot()
    finally:
        stop.set()
        for th in threads:
            th.join()

    assert not errors


def test_locked_timer_matches_timer_single_thread():
    # The subclass must not change semantics: same inputs -> identical SpanStats.
    base, locked = Timer(), LockedTimer()
    for tm in (base, locked):
        tm.enabled = True
        for ns in (100, 500, 200, 5_000_000):
            tm.add("x", ns)
    sb, sl = base.snapshot()["x"], locked.snapshot()["x"]
    assert (sl.calls, sl.total_ns, sl.min_ns, sl.max_ns) == (
        sb.calls, sb.total_ns, sb.min_ns, sb.max_ns,
    )
    assert sl.histo == sb.histo
    assert sl.grouped == sb.grouped
