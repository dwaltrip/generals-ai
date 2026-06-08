"""Bracket a training segment with the profiling bookend and an end-of-run report.

The BC entry points wrap their `bc_run` / `bc_resume` call in `instrumented_run`:
it opens the timing profiler when `--profile` is set and, on exit, writes a
markdown `report.md` for the run. Keeping the bookend here (the `bc` layer) lets
it import both `shared.timing_run` and `bc.run_report`; `shared/` can't, without
inverting the layering.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from training.bc.run_report import build_report
from training.shared.timing_run import begin, end_and_report


@contextmanager
def instrumented_run(run_dir: Path, profile: bool):
    """Profile bookend (when `profile`) + an always-on end-of-run `report.md`.

    When profiling, this also patches torch's pin-memory thread to time its
    inner loop (`bc.pin_instrument`), flushed to `prof/pin.json` for the report.

    `run_dir` must already exist — `begin()` mkdirs `<run_dir>/prof` under it.
    The report renders in `finally` so a partial one survives a crash, and it
    runs regardless of `--profile`: an unprofiled run still gets host draw,
    throughput, GPU util, and the bad-draw verdict; the producer obs-build
    section self-trims when there's no profiler summary.
    """
    prof_dir = run_dir / "prof"
    if profile:
        begin(prof_dir)
        # Lazy import: the module's top-level torch-version guard runs here
        # (only on profiled runs), and `enable` must precede DataLoader
        # iteration so the patched pin loop is in place when the thread spawns.
        from training.bc.pin_instrument import enable_pin_instrumentation

        enable_pin_instrumentation()
    try:
        yield
    finally:
        if profile:
            end_and_report(prof_dir)
            from training.bc.pin_instrument import flush as flush_pin

            flush_pin(prof_dir)
        _write_report(run_dir)


def _write_report(run_dir: Path) -> None:
    """Render `report.md` from the run's on-disk artifacts. Failure-isolated: a
    reporting error is printed, never raised, so it can't mask a training
    exception propagating through `instrumented_run`'s `finally`.
    """
    try:
        report_path = run_dir / "report.md"
        report_path.write_text(build_report(run_dir))
        print(f"wrote report: {report_path}")
    except Exception as exc:
        print(f"report generation failed (non-fatal): {exc!r}")
