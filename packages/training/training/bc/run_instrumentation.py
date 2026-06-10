"""Bracket a training segment with the profiling bookend and end-of-run reports.

The BC entry points wrap their `bc_run` / `bc_resume` call in `instrumented_run`:
it opens the timing profiler when `--profile` is set and, on exit, writes the
run's markdown reports — `quality.md` (training quality) and `perf.md` (perf /
starvation). Keeping the bookend here (the `bc` layer) lets it import both
`shared.timing_run` and `bc.perf_report`; `shared/` can't, without inverting
the layering.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from training.analysis.quality_report import build_quality_report
from training.bc.perf_report import build_perf_report
from training.shared.timing_run import begin, end_and_report


@contextmanager
def instrumented_run(run_dir: Path, profile: bool):
    """Profile bookend (when `profile`) + always-on end-of-run reports.

    When profiling, this also patches torch's pin-memory thread to time its
    inner loop (`bc.pin_instrument`), flushed to `prof/pin.json` for the report.

    `run_dir` must already exist — `begin()` mkdirs `<run_dir>/prof` under it.
    The reports render in `finally` so partial ones survive a crash, and they
    run regardless of `--profile`: an unprofiled run still gets the quality
    report plus host draw, throughput, GPU util, and the bad-draw verdict;
    the producer obs-build section self-trims when there's no profiler
    summary.
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
        _write_reports(run_dir)


def _write_reports(run_dir: Path) -> None:
    """Render `quality.md` and `perf.md` from the run's on-disk artifacts.
    Failure-isolated per report: a reporting error is printed, never raised,
    so it can't mask a training exception propagating through
    `instrumented_run`'s `finally`.
    """
    builders = [
        ("quality.md", build_quality_report),
        ("perf.md", build_perf_report),
    ]
    for name, builder in builders:
        try:
            report_path = run_dir / name
            report_path.write_text(builder(run_dir))
            print(f"wrote report: {report_path}")
        except Exception as exc:
            print(f"{name} generation failed (non-fatal): {exc!r}")
