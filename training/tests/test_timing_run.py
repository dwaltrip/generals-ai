"""Smoke coverage for the timing run-glue: FileSink + merge + report.

Each test resets the module-global `timer` (which `end_and_report` reads for the
main-process consumer tally) so cases don't bleed into each other.
"""

from __future__ import annotations

import json

import pytest

from shared.timing import timer
from shared.timing_run import FileSink, active_sink, begin, end_and_report


@pytest.fixture(autouse=True)
def _reset_global_timer():
    timer.reset()
    timer.enabled = False
    yield
    timer.reset()
    timer.enabled = False


def test_filesink_roundtrip_and_merge(tmp_path, capsys):
    sink = FileSink(tmp_path)
    # Two workers, same epoch, overlapping region names -> should sum. Snapshot
    # tuples are (ns, calls, grouped); a grouped=False region merges the same.
    sink.flush(1, 0, {"build_obs": (100, 2, True), "step_memory": (50, 2, True),
                      "encode_frame": (300, 2, False)})
    sink.flush(1, 1, {"build_obs": (200, 3, True), "encode_frame": (400, 3, False)})

    out = end_and_report(tmp_path)  # no main snapshot -> consumer empty

    assert out["producer"] == {
        "build_obs": (300, 5, True),
        "step_memory": (50, 2, True),
        "encode_frame": (700, 5, False),
    }
    assert out["consumer"] == {}

    # JSON round-trips tuples as lists.
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["producer"]["build_obs"] == [300, 5, True]

    printed = capsys.readouterr().out
    assert "producer" in printed and "build_obs" in printed


def test_begin_clears_stale_and_registers_sink(tmp_path):
    (tmp_path / "worker_9_9.json").write_text(json.dumps({"old": (1, 1)}))
    begin(tmp_path)
    assert not list(tmp_path.glob("*.json"))  # stale cleared
    assert active_sink() is not None
    assert timer.enabled is True


def test_consumer_table_from_main_timer(tmp_path):
    begin(tmp_path)  # enables the global timer
    with timer.section("fetch_wait"):
        pass

    out = end_and_report(tmp_path)

    assert set(out["consumer"]) == {"fetch_wait"}
    assert out["producer"] == {}
    assert (tmp_path / "main.json").exists()


def test_summary_excludes_itself_on_rerun(tmp_path):
    # summary.json must never be folded back into the producer sum.
    FileSink(tmp_path).flush(1, 0, {"build_obs": (100, 1, True)})
    end_and_report(tmp_path)  # writes summary.json
    out = end_and_report(tmp_path)  # second pass must not re-count summary.json
    assert out["producer"] == {"build_obs": (100, 1, True)}
