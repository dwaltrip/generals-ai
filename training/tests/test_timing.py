"""Unit coverage for the named-span Timer (`shared.timing`).

Tests use fresh `Timer()` instances rather than the module-global `timer`, so
each case is isolated from the process-global state.
"""

from __future__ import annotations

import pytest

from shared.timing import Timer, timer


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
    assert snap["sec"][1] == 2  # two recorded spans
    assert snap["man"][1] == 1
    assert snap["fn"][1] == 1
    assert all(ns >= 0 for ns, _ in snap.values())


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
    assert t.snapshot()["h2d"] == (8000, 2)


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
