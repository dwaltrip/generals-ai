"""
Background sampler for GPU utilization + memory, written to a JSONL sidecar.

## What it does

When training on CUDA, this spawns a daemon thread that wakes up once per
second and writes one JSONL record to a caller-specified sidecar file:

    {"t_sec": 1.0, "gpu_util_pct": 87, "mem_alloc_mb": 421, "mem_reserved_mb": 512,
     "cpu_steal_pct": 0.0, "load_avg_1m": 6.2}
    ...

`gpu_util_pct` is `null` when the NVML backend (`nvidia-ml-py`) is missing —
memory comes from torch directly, so those fields keep recording regardless.

`cpu_steal_pct` (time the hypervisor ran *other* tenants while we were
runnable) and `load_avg_1m` are host-contention signals: a slow draw that
isn't our own compute shows up as elevated steal here. `cpu_steal_pct` is
`null` on the first sample (it's a delta) and off-Linux.

`t_sec` is monotonic time since the sidecar started, so a `tail -f` on this
file alongside `run.log` aligns one-to-one in human time.

## Why a thread, not a subprocess

A daemon Python thread sampling at 1 Hz costs ~nothing (GIL impact at
sub-Hz sampling is negligible) and lives in the same process as training —
no extra Python interpreter, no IPC, automatically dies when training exits.
Subprocess (`nvidia-smi` polling) would also work; threads are simpler.

## What it doesn't do

- **Non-CUDA devices.** No-op on MPS/CPU. `torch.cuda.utilization()` only
  exists for CUDA, and MPS doesn't have a clean equivalent Python API.
- **Per-process accounting.** `gpu_util_pct` is the *device-wide* number.
  On Modal we have the box to ourselves so this is fine; on shared boxes
  it would be misleading.
- **Profiling-grade detail.** This answers "is the GPU busy?", not "doing
  what?". For the latter, use `torch.profiler.profile()` directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time

import torch


# TODO: CPU steal + load average are host-contention signals, not GPU metrics —
# they don't belong in a module named `gpu_sidecar`. If host sampling grows,
# rename this to something host-general (e.g. `resource_sidecar`) or split the
# host signals into their own sampler sharing the same daemon thread.
def _cpu_steal_total() -> tuple[int, int] | None:
    """(steal_ticks, total_ticks) from the aggregate `cpu` line of /proc/stat,
    or None off-Linux. Steal is field 8 (after the `cpu` label)."""
    try:
        fields = Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]
        nums = [int(x) for x in fields]
        steal = nums[7] if len(nums) > 7 else 0
        return steal, sum(nums)
    except (OSError, ValueError, IndexError):
        return None


def _sampler_loop(
    jsonl_path: Path,
    stop_event: threading.Event,
    sample_interval_sec: float,
) -> None:
    """Daemon-thread body. Opens the JSONL file, samples until stopped."""
    fp = jsonl_path.open("x", buffering=1)
    t0 = time.monotonic()
    util_err_logged = False
    prev_steal = _cpu_steal_total()  # delta baseline; steal_pct is null until 2nd sample
    try:
        # `stop_event.wait(timeout)` returns True if set, False on timeout.
        # Using it instead of `time.sleep` lets us exit immediately when the
        # context manager unwinds, instead of waiting out a full interval.
        while not stop_event.wait(sample_interval_sec):
            # Memory is torch-native (no NVML backend), so it's always
            # available on CUDA — sample it unconditionally.
            mem_alloc = torch.cuda.memory_allocated()
            mem_reserved = torch.cuda.memory_reserved()
            # Utilization goes through NVML (`nvidia-ml-py`). If that's
            # missing, degrade to null util and note it once — never let it
            # kill memory logging, which is what the old break-on-error did
            # (it dropped every record after the first failure).
            try:
                gpu_util_pct: int | None = int(torch.cuda.utilization())
            except Exception as exc:
                gpu_util_pct = None
                if not util_err_logged:
                    fp.write(json.dumps({"util_error": repr(exc)}) + "\n")
                    util_err_logged = True

            # Host-contention signals. Steal is a delta over the interval, so
            # the first sample (no baseline) emits null.
            cur_steal = _cpu_steal_total()
            steal_pct: float | None = None
            if cur_steal is not None and prev_steal is not None:
                d_steal = cur_steal[0] - prev_steal[0]
                d_total = cur_steal[1] - prev_steal[1]
                steal_pct = round(d_steal / d_total * 100, 1) if d_total > 0 else None
            if cur_steal is not None:
                prev_steal = cur_steal
            try:
                load_avg: float | None = round(os.getloadavg()[0], 2)
            except OSError:
                load_avg = None

            rec = {
                "t_sec": round(time.monotonic() - t0, 2),
                "gpu_util_pct": gpu_util_pct,
                "mem_alloc_mb": mem_alloc // (1024 * 1024),
                "mem_reserved_mb": mem_reserved // (1024 * 1024),
                "cpu_steal_pct": steal_pct,
                "load_avg_1m": load_avg,
            }
            fp.write(json.dumps(rec) + "\n")
    finally:
        fp.close()


@contextmanager
def gpu_util_sidecar(
    jsonl_path: Path,
    device: torch.device,
    sample_interval_sec: float = 1.0,
) -> Iterator[None]:
    """Run a 1 Hz GPU util/memory sampler for the duration of the block.

    On non-CUDA devices this is a no-op (yields immediately). On CUDA, a
    daemon thread writes JSONL samples to `jsonl_path` (opened exclusive-
    create) until the block exits.
    """
    if device.type != "cuda":
        # Quiet no-op: don't pollute the log with "skipped because MPS"
        # noise. Behavior is documented at the call site / in this module.
        yield
        return

    stop_event = threading.Event()
    thread = threading.Thread(
        target=_sampler_loop,
        args=(jsonl_path, stop_event, sample_interval_sec),
        name="gpu-util-sidecar",
        daemon=True,
    )
    thread.start()
    print(f"gpu-util sidecar started ({sample_interval_sec}s interval)")
    try:
        yield
    finally:
        stop_event.set()
        # Best-effort clean shutdown: wait briefly for the writer thread
        # to flush its final line. Daemon=True means we don't block exit
        # if the thread is wedged for some reason.
        thread.join(timeout=2.0)
