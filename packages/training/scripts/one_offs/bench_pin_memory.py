"""Micro-bench: split the pin-memory cost into alloc vs memcpy, fp32 vs fp16.

We measured the DataLoader pin thread spending ~233 ms/batch inside
`pin_memory()` — ~10x slower than a memcpy of that size should be. This probe
isolates WHY, on a real obs-batch-shaped tensor, with no pipeline involved:

  - pin_fp32      — `t.pin_memory()`: alloc (maybe cached) + copy. Today's cost.
  - copy_fp32     — `buf.copy_(t)` into a REUSED pre-pinned buffer: copy only.
                    pin − copy = the per-batch alloc / first-touch cost.
  - pin_fp32_shm  — same as pin_fp32 but the source is in shared memory (what
                    the real workers hand over). Slower than plain ⇒ the shm
                    read path matters.
  - *_fp16        — half the bytes. fp32/fp16 ratio = the per-byte fraction.
  - copy at 1 vs N threads — the real pin thread runs `set_num_threads(1)`; if
                    N threads is much faster, the copy is bandwidth-bound and
                    single-threading throttles it (→ multi-threaded pinning).
  - h2d / e2e     — confirm h2d is small, and time the proposed ring-buffer
                    path (reused pinned buffer → copy_ → non_blocking .to).

Reading it: copy ≈ pin ⇒ memcpy-bound (→ fp16/uint8, multi-thread). pin ≫ copy
⇒ alloc/first-touch-bound (→ pre-allocated pinned ring buffer). If NEITHER
variant here reproduces ~233 ms, the real cost is the cross-process / cross-NUMA
shm read that a single-process probe can't show — itself a useful narrowing.

The first run answered that: in isolation the copy is ~47 ms (n=20), ~5x faster
than the ~233 ms the live pin thread spends — alloc is free (the cache reuses),
shm backing doesn't matter, bandwidth is normal. So the slowdown is the *live
workload*. `bench_contention` tests the leading suspect — memory-bandwidth
contention — by timing the pinned copy while N background threads hammer memory,
mimicking the 8 workers' collate. If the copy slows toward ~233 ms as N→8,
contention is confirmed, and the dose-response sizes the fp16 payoff.

Run: uv run modal run packages/training/scripts/one_offs/bench_pin_memory.py
"""

from __future__ import annotations

import modal

from training.settings import TRAINING_REQS


image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_pip_install(requirements=[str(TRAINING_REQS)])
    .add_local_python_source("training", "settings", "utils")
)

app = modal.App("bc-train", image=image)


@app.function(gpu="H100", timeout=600)
def bench_pin(dense_history_n: int) -> dict:
    import os
    import time

    import torch

    from training.bc.constants import obs_channel_count

    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    torch.zeros(1, device=dev)  # init the CUDA context before any timing
    torch.cuda.synchronize()

    C = obs_channel_count(dense_history_n)
    B, H, W = 512, 32, 32
    shape = (B, C, H, W)

    # Real data (not zeros) so pages are materialized and no zero-page shortcut.
    src32 = torch.rand(shape, dtype=torch.float32)
    src32_shm = src32.clone().share_memory_()
    src16 = src32.half()
    buf32 = torch.empty(shape, dtype=torch.float32).pin_memory()
    buf16 = torch.empty(shape, dtype=torch.float16).pin_memory()
    bytes32 = src32.numel() * src32.element_size()
    bytes16 = src16.numel() * src16.element_size()

    def cold(fn) -> float:
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        return (time.perf_counter_ns() - t0) / 1e6

    def timeit(fn, warmup: int = 3, iters: int = 15) -> tuple[float, float]:
        for _ in range(warmup):
            fn()
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter_ns() - t0)
        ts.sort()
        return ts[0] / 1e6, ts[len(ts) // 2] / 1e6

    default_threads = torch.get_num_threads()
    n_multi = min(8, default_threads)
    results: dict[str, tuple[float, float]] = {}

    # True cold alloc (cache empty, context already warm) — the worst case.
    cold_pin_ms = cold(lambda: src32.pin_memory())

    torch.set_num_threads(1)  # match the real pin thread
    results["pin_fp32 (1t)"] = timeit(lambda: src32.pin_memory())
    results["pin_fp32_shm (1t)"] = timeit(lambda: src32_shm.pin_memory())
    results["copy_fp32 (1t)"] = timeit(lambda: buf32.copy_(src32))
    results["pin_fp16 (1t)"] = timeit(lambda: src16.pin_memory())
    results["copy_fp16 (1t)"] = timeit(lambda: buf16.copy_(src16))
    results["h2d_fp32 (1t)"] = timeit(lambda: buf32.to(dev, non_blocking=True))
    results["e2e copy+h2d (1t)"] = timeit(
        lambda: (buf32.copy_(src32), buf32.to(dev, non_blocking=True))
    )

    torch.set_num_threads(n_multi)
    results[f"copy_fp32 ({n_multi}t)"] = timeit(lambda: buf32.copy_(src32))
    results[f"pin_fp32 ({n_multi}t)"] = timeit(lambda: src32.pin_memory())
    torch.set_num_threads(default_threads)

    return {
        "dense_history_n": dense_history_n,
        "channels": C,
        "shape": list(shape),
        "MiB_fp32": round(bytes32 / 1024**2, 1),
        "MiB_fp16": round(bytes16 / 1024**2, 1),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "cpu_count": os.cpu_count(),
        "default_threads": default_threads,
        "cold_pin_fp32_ms": round(cold_pin_ms, 1),
        "bytes32": bytes32,
        "bytes16": bytes16,
        "results": {k: [round(v[0], 2), round(v[1], 2)] for k, v in results.items()},
    }


@app.function(gpu="H100", timeout=600)
def bench_contention(dense_history_n: int, load_levels: list[int]) -> dict:
    """Time the pinned copy while N background threads hammer memory. If the copy
    slows toward the live ~233 ms as N→8, memory-bandwidth contention is the
    cause. Each load thread copies its own ~150 MiB buffers in a tight loop
    (torch's CPU copy_ releases the GIL, so they generate real concurrent DRAM
    traffic), standing in for the workers' collate byte movement."""
    import threading
    import time

    import torch

    from training.bc.constants import obs_channel_count

    assert torch.cuda.is_available()
    torch.zeros(1, device="cuda")  # init CUDA context
    torch.cuda.synchronize()
    torch.set_num_threads(1)  # match the pin thread / container default

    C = obs_channel_count(dense_history_n)
    shape = (512, C, 32, 32)
    src = torch.rand(shape, dtype=torch.float32)
    buf = torch.empty(shape, dtype=torch.float32).pin_memory()  # reused pinned dst
    nbytes = src.numel() * src.element_size()

    # Background load buffers: ~150 MiB each, materialized (fill_) so reads hit DRAM.
    load_el = 150 * 1024**2 // 4

    def measure(iters: int = 20) -> tuple[float, float]:
        for _ in range(3):
            buf.copy_(src)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            buf.copy_(src)
            ts.append(time.perf_counter_ns() - t0)
        ts.sort()
        return ts[0] / 1e6, ts[len(ts) // 2] / 1e6

    out: dict[int, list[float]] = {}
    for n_load in load_levels:
        stop = threading.Event()
        pairs = [
            (torch.empty(load_el).fill_(1.0), torch.empty(load_el))
            for _ in range(n_load)
        ]

        def loadfn(s, d, stop) -> None:
            while not stop.is_set():
                d.copy_(s)

        threads = [
            threading.Thread(target=loadfn, args=(s, d, stop), daemon=True)
            for s, d in pairs
        ]
        for t in threads:
            t.start()
        time.sleep(0.3)  # let the load ramp up
        mn, med = measure()
        stop.set()
        for t in threads:
            t.join()
        out[n_load] = [round(mn, 2), round(med, 2)]
        del pairs

    return {
        "dense_history_n": dense_history_n,
        "channels": C,
        "MiB": round(nbytes / 1024**2, 1),
        "load_buf_MiB": 150,
        "gpu": torch.cuda.get_device_name(),
        "nbytes": nbytes,
        "copy_ms_by_load": {str(k): v for k, v in out.items()},
    }


@app.local_entrypoint()
def main() -> None:
    for n in (5, 20):
        r = bench_pin.remote(n)
        print(f"\n=== n={r['dense_history_n']}  ({r['channels']} ch, "
              f"{r['MiB_fp32']} MiB fp32 / {r['MiB_fp16']} MiB fp16) ===")
        print(f"  {r['gpu']} · torch {r['torch']} · "
              f"{r['cpu_count']} cores · default_threads={r['default_threads']}")
        print(f"  cold first pin_fp32: {r['cold_pin_fp32_ms']} ms")
        print(f"  {'op':22s} {'min ms':>8s} {'med ms':>8s} {'GiB/s (min)':>12s}")
        for name, (mn, med) in r["results"].items():
            nbytes = r["bytes16"] if "fp16" in name else r["bytes32"]
            gibs = nbytes / (mn * 1e-3) / 1024**3 if mn > 0 else 0.0
            print(f"  {name:22s} {mn:8.2f} {med:8.2f} {gibs:12.2f}")

    # Contention dose-response (the live ~233 ms is ~5x the isolated copy).
    c = bench_contention.remote(20, [0, 2, 4, 8, 12, 16])
    base = c["copy_ms_by_load"]["0"][0]
    print(f"\n=== contention: pinned copy under N memory-load threads "
          f"(n=20, {c['MiB']} MiB, {c['load_buf_MiB']} MiB/thread) ===")
    print(f"  {'load threads':>12s} {'min ms':>8s} {'med ms':>8s} "
          f"{'GiB/s':>8s} {'vs N=0':>8s}")
    for k in sorted(c["copy_ms_by_load"], key=int):
        mn, med = c["copy_ms_by_load"][k]
        gibs = c["nbytes"] / (mn * 1e-3) / 1024**3 if mn > 0 else 0.0
        print(f"  {k:>12s} {mn:8.2f} {med:8.2f} {gibs:8.2f} {mn / base:7.1f}x")
