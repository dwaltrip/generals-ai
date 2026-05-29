"""
Background sampler for GPU utilization + memory, written to a JSONL sidecar.

## What it does

When training on CUDA, this spawns a daemon thread that wakes up once per
second and writes one JSONL record to a caller-specified sidecar file:

    {"t_sec": 1.0, "gpu_util_pct": 87, "mem_alloc_mb": 421, "mem_reserved_mb": 512}
    {"t_sec": 2.0, "gpu_util_pct": 93, "mem_alloc_mb": 421, "mem_reserved_mb": 512}
    ...

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

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch


def _sampler_loop(
    jsonl_path: Path,
    stop_event: threading.Event,
    sample_interval_sec: float,
) -> None:
    """Daemon-thread body. Opens the JSONL file, samples until stopped."""
    fp = jsonl_path.open("x", buffering=1)
    t0 = time.monotonic()
    try:
        # `stop_event.wait(timeout)` returns True if set, False on timeout.
        # Using it instead of `time.sleep` lets us exit immediately when the
        # context manager unwinds, instead of waiting out a full interval.
        while not stop_event.wait(sample_interval_sec):
            try:
                gpu_util_pct = torch.cuda.utilization()
                mem_alloc = torch.cuda.memory_allocated()
                mem_reserved = torch.cuda.memory_reserved()
            except Exception as exc:
                # If polling fails (e.g. pynvml issue), log once and stop.
                fp.write(json.dumps({"error": repr(exc)}) + "\n")
                break

            rec = {
                "t_sec": round(time.monotonic() - t0, 2),
                "gpu_util_pct": int(gpu_util_pct),
                "mem_alloc_mb": mem_alloc // (1024 * 1024),
                "mem_reserved_mb": mem_reserved // (1024 * 1024),
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
