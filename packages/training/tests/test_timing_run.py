"""Smoke coverage for the timing run-glue: FileSink + merge + report.

Each test resets the module-global `timer` (which `end_and_report` reads for the
main-process consumer tally) so cases don't bleed into each other.
"""

from __future__ import annotations

import json

import pytest

from shared.timing import SpanStats, timer
from shared.timing_run import FileSink, _span_from_json, active_sink, begin, end_and_report


@pytest.fixture(autouse=True)
def _reset_global_timer():
    timer.reset()
    timer.enabled = False
    yield
    timer.reset()
    timer.enabled = False


def _make_span(total_ns: int, calls: int, grouped: bool = True, **kw) -> SpanStats:
    return SpanStats(total_ns=total_ns, calls=calls, grouped=grouped, **kw)


def test_filesink_roundtrip_and_merge(tmp_path, capsys):
    sink = FileSink(tmp_path)
    sink.flush(1, 0, {
        "build_obs": _make_span(100, 2, min_ns=40, max_ns=60, histo={2: 2}),
        "step_memory": _make_span(50, 2, min_ns=20, max_ns=30, histo={1: 2}),
        "encode_frame": _make_span(300, 2, grouped=False, min_ns=100, max_ns=200, histo={4: 2}),
    })
    sink.flush(1, 1, {
        "build_obs": _make_span(200, 3, min_ns=30, max_ns=100, histo={2: 1, 3: 2}),
        "encode_frame": _make_span(400, 3, grouped=False, min_ns=80, max_ns=250, histo={4: 3}),
    })

    out = end_and_report(tmp_path)  # no main snapshot -> consumer empty

    prod = out["producer"]
    bo = _span_from_json(prod["build_obs"])
    assert bo.total_ns == 300
    assert bo.calls == 5
    assert bo.grouped is True
    assert bo.min_ns == 30
    assert bo.max_ns == 100
    assert bo.histo == {2: 3, 3: 2}

    sm = _span_from_json(prod["step_memory"])
    assert sm.total_ns == 50
    assert sm.calls == 2

    ef = _span_from_json(prod["encode_frame"])
    assert ef.total_ns == 700
    assert ef.calls == 5
    assert ef.grouped is False
    assert ef.min_ns == 80
    assert ef.max_ns == 250
    assert ef.histo == {4: 5}

    assert out["consumer"] == {}

    # JSON round-trips correctly.
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["producer"]["build_obs"]["total_ns"] == 300

    printed = capsys.readouterr().out
    assert "producer" in printed and "build_obs" in printed


def test_begin_clears_stale_and_registers_sink(tmp_path):
    (tmp_path / "worker_9_9.json").write_text(json.dumps({"old": [1, 1]}))
    begin(tmp_path)
    assert not list(tmp_path.glob("*.json"))  # stale cleared
    assert active_sink() is not None
    assert timer.enabled is True


def test_consumer_table_from_main_timer(tmp_path):
    begin(tmp_path)  # enables the global timer
    with timer.section("fetch_wait"):
        pass

    out = end_and_report(tmp_path)

    assert "fetch_wait" in out["consumer"]
    assert out["producer"] == {}
    assert (tmp_path / "main.json").exists()


def test_summary_excludes_itself_on_rerun(tmp_path):
    # summary.json must never be folded back into the producer sum.
    sink = FileSink(tmp_path)
    sink.flush(1, 0, {"build_obs": _make_span(100, 1)})
    end_and_report(tmp_path)  # writes summary.json
    out = end_and_report(tmp_path)  # second pass must not re-count summary.json
    bo = _span_from_json(out["producer"]["build_obs"])
    assert bo.total_ns == 100


def test_span_from_json_legacy_list():
    s = _span_from_json([1000, 5, True])
    assert s.total_ns == 1000
    assert s.calls == 5
    assert s.grouped is True
    assert s.histo == {}
    assert s.min_ns is None

    s2 = _span_from_json([2000, 3])
    assert s2.grouped is True


def test_span_from_json_dict():
    s = _span_from_json({
        "total_ns": 500, "calls": 2, "grouped": False,
        "min_ns": 200, "max_ns": 300,
        "histo": {"3": 1, "5": 1},
    })
    assert s.total_ns == 500
    assert s.grouped is False
    assert s.histo == {3: 1, 5: 1}  # keys coerced to int


def test_merge_min_max_across_workers(tmp_path):
    sink = FileSink(tmp_path)
    sink.flush(1, 0, {"x": _make_span(100, 1, min_ns=50, max_ns=50, histo={2: 1})})
    sink.flush(1, 1, {"x": _make_span(200, 1, min_ns=10, max_ns=200, histo={2: 1, 4: 1})})
    # Worker 1 has a None min/max span (no histo data either — legacy-like)
    sink.flush(1, 2, {"x": _make_span(300, 1)})

    out = end_and_report(tmp_path)
    s = _span_from_json(out["producer"]["x"])
    assert s.min_ns == 10
    assert s.max_ns == 200
    assert s.histo == {2: 2, 4: 1}
    assert s.total_ns == 600
    assert s.calls == 3
