"""Instrument PyTorch's DataLoader pin-memory thread to localize the n=20 stall.

The dense-history n=20 slowdown is GPU data-starvation, bracketed by elimination
to the worker→main handoff machinery — and the single background `pin_memory`
thread is the prime suspect, but it's the one stretch of the pipeline we can't
seam from our own code (it lives inside torch). This module monkeypatches
torch's `_pin_memory_loop` with a faithful copy that times each step, turning
that blind spot into measurement. Background and the competing hypotheses:
`docs/pytorch-dataloader-pin-memory-bottleneck.md`.

Each iteration of the pin thread's loop is split into four spans, and their
relative sizes arbitrate the hypotheses directly:

  - **pin_get**     — blocked waiting for a batch from the workers (result
                      queue). Large ⇒ the pin thread is *starved*; the
                      bottleneck is upstream (workers), not here.
  - **pin_copy**    — `pin_memory()`: the page-locked copy (cudaHostAlloc +
                      memcpy). Large ⇒ copy/alloc-bound (→ fp16/uint8 obs, or a
                      pre-allocated pinned ring buffer).
  - **pin_destruct**— freeing the source shared-memory batch (munmap, under the
                      GIL). Large ⇒ the GIL-contention mechanism (→ fewer/bigger
                      lifecycle events, smaller regions, free-threaded Python).
  - **pin_put**     — blocked handing the pinned batch to the main loop (data
                      queue). Large ⇒ the *consumer* (main loop / GPU) is the
                      limit — a healthy GPU-bound run, not a pin stall.

`pin_copy + pin_destruct` is the pin thread's own serial work; compared against
the batch period it answers "is the pin thread the throughput ceiling?". The
spans also carry the usual histograms, so each one's `max` reveals whether the
rare multi-second freezes live in (say) `pin_destruct`.

Version coupling: the instrumented loop is a re-vendored copy of a specific
torch version's `_pin_memory_loop`. The import-time guard below hard-aborts if
the running torch differs, so a silent drift can't make us measure a stale copy.
"""

from __future__ import annotations

import json
from pathlib import Path
import queue
import time

import torch
from torch._utils import ExceptionWrapper
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch.utils.data._utils.pin_memory import pin_memory

from training.shared.timing import LockedTimer
from training.shared.timing_run import _tally_to_json


# The torch version whose `_pin_memory_loop` `_do_one_step` below mirrors. The
# loop's structure (the `do_one_step` helper, the timeout-retry put) is version-
# specific, so we pin it and check at import — before any patching or measuring.
_EXPECTED_TORCH = "2.12.0"
_running_torch = torch.__version__.split("+")[0]  # strip the +cuXXX build suffix
if _running_torch != _EXPECTED_TORCH:
    raise RuntimeError(
        f"training.bc.pin_instrument vendors torch {_EXPECTED_TORCH}'s _pin_memory_loop; "
        f"running torch is {torch.__version__}. Re-vendor _pin_memory_loop from "
        f"the new version and bump _EXPECTED_TORCH before profiling."
    )


# The pin thread runs on a background thread in the main process, so its timer
# must be thread-safe (the worker/main timers are single-threaded and aren't).
pin_timer = LockedTimer()

# Set by `enable_pin_instrumentation` so the swap is idempotent / inspectable.
_original_loop = None


def _do_one_step(in_queue, out_queue, device, device_id, done_event) -> None:
    """One iteration of the pin loop, timed. A faithful copy of torch 2.12.0's
    inner `do_one_step`, with four timing seams and an explicit source-tensor
    delete to isolate the destruction cost. Pulled out as a module-level
    function so it can be unit-tested without spawning the thread."""
    # pin_get — wait for a batch from a worker. Timed on both paths so the
    # get-time accounting stays complete even when the thread is idle-polling.
    t0 = time.perf_counter_ns()
    try:
        r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
    except queue.Empty:
        pin_timer.add("pin_get", time.perf_counter_ns() - t0)
        return
    pin_timer.add("pin_get", time.perf_counter_ns() - t0)

    idx, data = r
    if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
        # pin_copy — the page-locked copy, recursing over the batch dict.
        t1 = time.perf_counter_ns()
        try:
            data = pin_memory(data, device)
        except Exception:
            data = ExceptionWrapper(
                where=f"in pin memory thread for device {device_id}"
            )
        pin_timer.add("pin_copy", time.perf_counter_ns() - t1)

        # pin_destruct — free the SOURCE shared-memory batch. After the copy,
        # `data` points at the pinned copy, so the tuple `r` holds the only
        # remaining reference to the original shm tensors; `del r` drops it,
        # firing their destructors (munmap of the shm regions) right here where
        # we can time it, instead of implicitly at the `r = (idx, data)` rebind
        # below. Behavior-preserving — torch drops the same tuple anyway — and
        # it isolates the GIL-held-munmap mechanism from the queue waits.
        # (`idx` is a separate int reference, unaffected by `del r`.)
        t2 = time.perf_counter_ns()
        del r
        pin_timer.add("pin_destruct", time.perf_counter_ns() - t2)

        r = (idx, data)
    # else: shutting down, or the worker already wrapped an exception — forward
    # `r` unchanged (its `data` must stay alive); record no copy/destruct.

    # pin_put — hand the pinned batch to the main loop. Large ⇒ consumer-bound.
    t3 = time.perf_counter_ns()
    while not done_event.is_set():
        try:
            out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
            break
        except queue.Full:
            continue
    pin_timer.add("pin_put", time.perf_counter_ns() - t3)


def _pin_memory_loop_instrumented(
    in_queue, out_queue, device_id, done_event, device
) -> None:
    """Drop-in replacement for torch's `_pin_memory_loop` (same signature) that
    routes each iteration through the timed `_do_one_step`. The preamble mirrors
    torch 2.12.0's verbatim."""
    torch.set_num_threads(1)
    torch.multiprocessing._set_thread_name("pt_data_pin")
    torch.accelerator.set_device_index(device_id)
    while not done_event.is_set():
        _do_one_step(in_queue, out_queue, device, device_id, done_event)


def enable_pin_instrumentation() -> None:
    """Swap torch's pin-memory loop for the instrumented copy and arm the timer.

    Must run before the DataLoader iterator is created: the iterator spawns the
    pin thread with `target=_utils.pin_memory._pin_memory_loop`, a module-
    attribute lookup at spawn time, so patching the attribute beforehand takes
    effect. Idempotent — re-enabling keeps the saved original.
    """
    global _original_loop
    import torch.utils.data._utils.pin_memory as _pm

    if _original_loop is None:
        _original_loop = _pm._pin_memory_loop
    _pm._pin_memory_loop = _pin_memory_loop_instrumented
    pin_timer.enabled = True


def flush(prof_dir: Path) -> None:
    """Write the pin-thread tally to `<prof_dir>/pin.json`, in the same
    `SpanStats` schema as the worker/main timing files (so `perf_report` reads it
    with the existing deserializer). The snapshot is taken under the timer's
    lock, so it's safe to call while the pin thread is still alive."""
    snap = pin_timer.snapshot()
    if snap:
        (prof_dir / "pin.json").write_text(json.dumps(_tally_to_json(snap)))
