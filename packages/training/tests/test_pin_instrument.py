"""Unit coverage for the pin-thread instrument (`bc.pin_instrument`).

These exercise `_do_one_step` — the timed inner loop — directly, with a stubbed
`pin_memory` so they need no CUDA. The behavioral claims under test are: the
batch is forwarded unchanged, the four spans are recorded, and the `del r`
reorder actually frees the source batch (the destruction-isolation trick).
"""

from __future__ import annotations

import queue
import threading

import pytest
from torch._utils import ExceptionWrapper

from training.bc import pin_instrument


@pytest.fixture(autouse=True)
def _fresh_pin_timer():
    """Arm + clear the module-global pin_timer around each test."""
    pin_instrument.pin_timer.enabled = True
    pin_instrument.pin_timer.reset()
    yield
    pin_instrument.pin_timer.reset()
    pin_instrument.pin_timer.enabled = False


def _step(in_items, monkeypatch, *, pin=None):
    """Run one `_do_one_step` over a queue pre-filled with `in_items`, returning
    the out_queue. `pin` overrides the stubbed pin_memory (default: a fresh
    object per call, so the source's last reference is the loop's tuple)."""
    monkeypatch.setattr(
        pin_instrument, "pin_memory", pin or (lambda data, device: object())
    )
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    done = threading.Event()
    for item in in_items:
        in_q.put(item)
    pin_instrument._do_one_step(in_q, out_q, None, 0, done)
    return out_q


def test_forwards_and_records_spans(monkeypatch):
    pinned = object()
    out_q = _step([(7, object())], monkeypatch, pin=lambda data, device: pinned)

    idx, data = out_q.get_nowait()
    assert idx == 7
    assert data is pinned  # the pinned copy was forwarded, not the source

    snap = pin_instrument.pin_timer.snapshot()
    assert {"pin_get", "pin_copy", "pin_destruct", "pin_put"} <= set(snap)
    assert all(snap[name].calls == 1 for name in ("pin_copy", "pin_destruct", "pin_put"))


def test_destruct_frees_source(monkeypatch):
    # The `del r` must drop the source batch's last reference. Track it via a
    # __del__ side effect; the test holds only the flag, never the object.
    flag = [False]

    class _Tracked:
        def __del__(self) -> None:
            flag[0] = True

    out_q = _step([(0, _Tracked())], monkeypatch)  # pin returns a fresh object()

    assert flag[0] is True  # source destroyed at `del r`, within pin_destruct
    assert pin_instrument.pin_timer.snapshot()["pin_destruct"].calls == 1
    out_q.get_nowait()  # batch still forwarded


def test_exception_wrapper_passthrough(monkeypatch):
    # An already-wrapped exception skips pinning and is forwarded unchanged.
    ew = ExceptionWrapper(where="test")
    out_q = _step([(3, ew)], monkeypatch)

    idx, data = out_q.get_nowait()
    assert idx == 3
    assert data is ew
    snap = pin_instrument.pin_timer.snapshot()
    assert "pin_copy" not in snap  # copy/destruct skipped on the wrapped path
    assert "pin_destruct" not in snap
    assert snap["pin_get"].calls == 1
    assert snap["pin_put"].calls == 1


def test_empty_queue_records_only_get(monkeypatch):
    # An empty poll records the get-wait, then bails — no copy/put. Shorten the
    # get timeout so the empty path returns fast.
    monkeypatch.setattr(pin_instrument, "pin_memory", lambda data, device: object())
    monkeypatch.setattr(pin_instrument, "MP_STATUS_CHECK_INTERVAL", 0.01)
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    done = threading.Event()
    pin_instrument._do_one_step(in_q, out_q, None, 0, done)

    snap = pin_instrument.pin_timer.snapshot()
    assert snap["pin_get"].calls == 1
    assert "pin_copy" not in snap
    assert out_q.empty()
