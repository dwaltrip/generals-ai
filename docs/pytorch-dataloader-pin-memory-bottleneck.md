# PyTorch DataLoader pin_memory bottleneck — reference

Reference doc on a known PyTorch architectural limitation: the single-threaded `pin_memory` loop in the DataLoader, how it becomes a throughput ceiling, and what options exist for working around it.

Written during investigation of the n=20 dense-history data starvation ([`6.05-3`](2026-06/6.05-3-obs-pipeline-starvation-n20.md), [`6.06-4`](2026-06/6.06-4-train-data-flow-profile-summary.md)). The findings here are general to any PyTorch DataLoader workload where the host→GPU data path is the bottleneck.

## Background: what "pinning" means

The GPU's DMA engine needs a stable physical address to copy from. Normally the OS can relocate or swap out pages at any time. "Pinning" (page-locking) tells the OS to lock specific pages in physical RAM so the GPU can DMA directly from them. Without pinning, CUDA must first copy data to an internal pinned staging buffer before initiating the DMA — that extra copy makes `pin_memory=False` dramatically slower.

In our case (512×126×H×W fp32 obs batches on H100), the sweep measured ~4 ms h2d with pinning vs ~200 ms without. (The "fp16" in our training config is model precision via autocast; the obs tensors going through the pipeline are fp32.)

## The DataLoader pipeline with pin_memory=True

With `num_workers > 0` and `pin_memory=True`, the DataLoader's data path is:

```
worker process                    main process
─────────────                    ────────────
__getitem__() per sample
        ↓
collate_fn() → batch tensor
        ↓
torch.stack into shared memory
        ↓
    result_queue  ──(IPC, shm handles)──→  pin_memory_loop (single bg thread)
                                                    ↓
                                           tensor.pin_memory()
                                           (cudaHostAlloc + memcpy)
                                                    ↓
                                               data_queue
                                                    ↓
                                           next(dataloader) in training loop
                                                    ↓
                                           tensor.to(device)  (DMA from pinned mem)
```

The critical architectural fact: **`_pin_memory_loop` is a single Python thread** in the main process. All output from all workers funnels through it. It is the only path from worker output to GPU-ready data.

## Two bottleneck mechanisms

### 1. Serial memcpy throughput

The pin_memory thread calls `tensor.pin_memory()` on every batch, which internally does `cudaHostAlloc` (allocate page-locked memory) + `memcpy` (copy from shared memory to the pinned buffer). With large batches and many workers producing data, this single thread can become the throughput ceiling.

PyTorch maintainer albanD acknowledged this and noted "it should be relatively simple to have multiple pinning threads in the main process to speed it up when you have a lot of worker threads" — but this has not been implemented in PyTorch's DataLoader.

Source: [Why not multiprocess pin_memory in data loader?](https://discuss.pytorch.org/t/why-not-multiprocess-pin-memory-in-data-loader/197345)

### 2. GIL contention from tensor destruction

When the pin_memory thread receives tensors from workers via shared memory, the old tensor references (the unpinned shared-memory copies) go out of scope. Their destructors trigger `munmap` syscalls to unmap the shared memory regions. The pin thread holds the Python GIL during these destructors, **freezing the main training thread**. This has been measured to produce **50+ ms GPU bubbles** on 8×A100 setups.

This mechanism is distinct from the raw memcpy throughput — the thread isn't slow at copying, it's blocking the training loop via GIL contention during cleanup.

Sources:
- [Significant perf reduction on Python GIL contention with dataloader pinning thread](https://github.com/pytorch/pytorch/issues/77139) — the 50+ ms bubble measurement
- [GIL contention between pin_memory & main trainer threads](https://discuss.pytorch.org/t/gil-contention-between-pin-memory-main-trainer-threads/198761) — additional discussion, albanD's proposed long-term fixes
- [Why is DataLoader slower when pin_memory=True](https://discuss.pytorch.org/t/why-is-dataloader-slower-when-pin-memory-true/126620) — profiling showing more time in thread lock acquisition with pin_memory=True
- [Three memory copies of every dataloader cpu tensor](https://github.com/pytorch/pytorch/issues/78018) — documents the full copy chain

### Which mechanism applies when

The serial-memcpy bottleneck scales with data volume: larger tensors or more workers producing data → more bytes to copy per second → the single thread saturates. The GIL-contention mechanism scales with the rate of tensor lifecycle events (allocations and deallocations of shared-memory regions) and may explain tail-latency spikes more than steady-state throughput loss. These two mechanisms aren't mutually exclusive — a workload can hit both.

In our case, either or both could contribute. The n=20 observation tensor is ~31% larger than n=5 (126 vs 96 channels), increasing both the memcpy volume and the size of shared-memory regions being mapped/unmapped. We haven't isolated which mechanism dominates.

A back-of-envelope check suggests raw memcpy throughput alone is not the bottleneck: at ~252 MiB/batch (n=20, fp32) and ~10–15 GB/s single-core memcpy, the copy takes ~17–25 ms. Our target is ~4 batches/sec (2084 sps / 512), so the pin thread's time budget is ~250 ms/batch — well above the raw copy cost. This points toward overhead beyond the memcpy itself: `cudaHostAlloc` page-locking, GIL contention during tensor destruction, or queue synchronization. (Caveat: `cudaHostAlloc` overhead is not captured in the memcpy estimate and could be substantial.)

## Pre-pinning in workers: not viable

The intuitive idea — have workers allocate pinned memory directly, skip the pin_memory thread — is blocked by a fundamental constraint.

**Shared memory and pinned memory are incompatible.** Workers communicate with the main process via shared memory (`multiprocessing.Queue` backed by `shm_open`/`mmap`). `cudaHostAlloc` (which `pin_memory()` uses) allocates a *new* page-locked buffer — it cannot pin existing shared memory regions. Even if a worker called `pin_memory()`, the result would be a non-shared allocation that cannot be sent back through the IPC queue. albanD confirmed this directly: "it is not possible to have memory that is both shared and pinned."

Source: [Why not multiprocess pin_memory in data loader?](https://discuss.pytorch.org/t/why-not-multiprocess-pin-memory-in-data-loader/197345)

**CUDA context issues in forked workers.** Beyond the shared/pinned incompatibility, `pin_memory()` requires an initialized CUDA context (`cudaHostAlloc` needs one). DataLoader workers are forked by default on Linux. CUDA contexts cannot be safely inherited across `fork` — this produces crashes, wrong-device allocations, and deadlocks.

Sources:
- [GPU 0 context created on GPU 1 worker](https://github.com/pytorch/pytorch/issues/58626)
- [Cannot re-initialize CUDA in forked subprocess](https://github.com/pytorch/pytorch/issues/40403)
- [Dataloader hangs with fork and pin_memory=True](https://github.com/pytorch/pytorch/issues/130610)

**In-place pinning (`cudaHostRegister`).** There is an open feature request for `pin_memory_()` (in-place) using `cudaHostRegister` instead of `cudaHostAlloc`. This could theoretically pin shared memory in-place, but the issue notes that `cudaHostRegister` has high overhead ("typically as slow or slower than copying un-pinned memory to the GPU") and only pays off for long-lived allocations. Not implemented as of mid-2026.

Source: [Feature request: pin_memory_()](https://github.com/pytorch/pytorch/issues/32167)

## Alternative libraries

These replace the DataLoader pipeline rather than patching the pin_memory thread.

### NVIDIA DALI

Moves data decoding and augmentation to the GPU. Manages its own pinned memory pools (using `cudaMallocHost` with stream-aware allocation) instead of relying on PyTorch's `pin_memory()`. Supports GPUDirect Storage for a direct data path from disk to GPU memory, bypassing CPU pinned memory entirely. Heavyweight dependency; designed for image/video workloads.

Docs: [DALI user guide](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html), [performance tuning](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/advanced_topics_performance_tuning.html)

### FFCV

Uses **threads instead of processes**, eliminating the shared-memory IPC layer entirely. Threads share the same address space, so there is no shared-memory-to-pinned-memory copy step. Uses pre-allocated pinned memory buffers and JIT-compiles the augmentation pipeline. Requires data in FFCV's custom `.beton` binary format.

Docs: [FFCV docs](https://docs.ffcv.io/), [bottleneck doctor](https://docs.ffcv.io/bottleneck_doctor.html), [parameter tuning](https://docs.ffcv.io/parameter_tuning.html)

### SPDL (Meta, 2025)

Thread-based data loading explicitly designed around the GIL problem. Data preprocessing functions release the GIL, enabling true concurrency. Reports 74% faster than PyTorch DataLoader on ImageNet, 38% less CPU, 50 GB less memory. Has experimental support for free-threaded Python 3.13t (no GIL).

Sources: [SPDL blog post](https://ai.meta.com/blog/spdl-faster-ai-model-training-with-thread-based-data-loading-reality-labs/), [SPDL paper](https://arxiv.org/abs/2504.20067)

### TensorDict (torchrl)

The closest to a targeted fix within the existing PyTorch ecosystem. `TensorDict.to()` has a `non_blocking_pin` option that spawns **multiple threads** to pin tensors in parallel before calling `to(device)`. This is essentially the "multiple pinning threads" approach albanD suggested but that PyTorch never shipped internally.

Source: [PyTorch pin_memory tutorial](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)

## Other workarounds

**CUDA-stream prefetcher.** Wraps the DataLoader iterator, uses a separate CUDA stream to overlap `tensor.to(device, non_blocking=True)` with the current training step on the default stream. Does not bypass the pin_memory thread, but hides h2d transfer latency behind GPU compute. Shipped in TorchTNT as `CudaDataPrefetcher`; PyTorch Geometric has `PrefetchLoader`. Many training codebases implement a lightweight version. Source: [TorchTNT CudaDataPrefetcher](https://meta-pytorch.org/tnt/stable/utils/generated/torchtnt.utils.data.CudaDataPrefetcher.html)

**Pre-allocated pinned buffers.** Allocate a fixed pinned tensor once (`torch.empty(...).pin_memory()`), then `buf.copy_(batch)` each iteration followed by `buf.to(device, non_blocking=True)`. Avoids repeated `cudaHostAlloc` calls. This is what FFCV does internally. Not a common pattern in user code but discussed on forums.

**Free-threaded Python (3.13t).** With the GIL removed, the pin_memory thread and training thread truly run concurrently, eliminating the GIL-contention mechanism. NVIDIA testing showed thread-based DataLoader workers significantly outperform process-based workers on free-threaded Python. Experimental as of mid-2026. Source: [NVIDIA blog on improved data loading with threads](https://developer.nvidia.com/blog/improved-data-loading-with-threads/)

**Reduce data volume through the handoff path.** Less data per batch means less work for the pin_memory thread. Options: smaller dtype (fp16 obs instead of fp32), fewer channels, compute some features on-GPU instead of in workers. This doesn't fix the architectural bottleneck but moves the crossover point — the same single thread can handle higher batch rates if each batch is smaller.

## What our sweep established

The 2026-06-07 config sweep (N × prefetch_factor × pin_memory, 8 cells, 800 batches each on H100; data in [`training/data/sweeps/2026-06-07-data-starve-handoff-probe/`](../training/data/sweeps/2026-06-07-data-starve-handoff-probe/)) confirmed:

- **pin_memory=false** is catastrophic: h2d goes from ~4 ms to ~200 ms per batch, halving throughput. Pinning is essential.
- **Workers are ~50% idle** at both n=5 and n=20 (handoff − obs_build_total ≈ 50% of handoff time). The 19% obs-build compute increase at n=20 is absorbed by idle headroom — workers are not the bottleneck.
- **prefetch_factor 2 vs 4** makes no difference at n=20 (2084 vs 2009 sps). Rules out a depth/latency issue — more buffering doesn't help because the throughput ceiling is upstream of the buffer.
- **12 workers** (tested separately) didn't help either. More producers doesn't increase throughput.

All three tests point the same direction: a throughput ceiling somewhere in the single-threaded handoff path between workers and GPU. The pin_memory thread is the prime candidate (it's the only serialization point in that path), but we have not directly instrumented it to confirm.

The sweep also showed multi-second tail stalls in the `handoff` span (max 2.6–3.1 s across pinned configs, while p50 is ~1.8 µs). The cause of these tails is unknown — candidates include GIL contention (tensor-destruction `munmap` pauses), OS scheduling, GC, or shared-memory contention.

See [`6.06-4`](2026-06/6.06-4-train-data-flow-profile-summary.md) for the full pipeline map with observability status of each stage. Section B ("the handoff machinery") is the uninstrumented region where the bottleneck sits.

**Relationship to earlier findings.** [`6.05-3`](2026-06/6.05-3-obs-pipeline-starvation-n20.md) ruled out "Consumer (pin_memory / H2D)" with the rationale "pin thread far faster than the batch rate," and hypothesized memory bandwidth as the residual bottleneck (leading lever: fp16 obs). The sweep evidence above reopens the handoff path: workers are 50% idle (their 19% compute increase fits well within the idle budget, so they're not memory-bandwidth-bound), yet more workers and more prefetch don't help — the ceiling is downstream of the workers, not in their compute. The pin_memory thread / handoff machinery is the revised leading suspect. fp16 obs remains a useful lever either way (it halves bytes through the handoff path regardless of which specific mechanism is binding).

## What we haven't confirmed

- Whether it's specifically the `pin_memory()` memcpy, the GIL contention during tensor destruction, the result queue IPC, or some combination. We've bracketed it to the handoff path but haven't isolated which stage within it.
- Whether the multi-second handoff tail stalls are GIL-related or something else (GC, OS scheduling, shared-memory contention).
- Whether any of the alternative libraries (DALI, FFCV, SPDL, TensorDict) would help in our specific case — our data pipeline is custom (game replay parsing → obs tensor construction), not a standard image-loading workload.

## Potential next steps for our project

Roughly ordered by effort:

1. **Instrument the pin_memory thread.** Monkey-patch `torch.utils.data._utils.pin_memory._pin_memory_loop` to time the `pin_memory()` call vs queue get/put. Would directly answer "is it the pinning copy or the queue or the GIL?" Low effort, high signal.
2. **fp16 obs tensors.** Halves the data volume through the entire handoff path (already identified as the leading lever in [`6.05-3`](2026-06/6.05-3-obs-pipeline-starvation-n20.md)). Doesn't fix the architectural bottleneck but moves the crossover point — the pin_memory thread has half as many bytes to copy.
3. **TensorDict's multi-threaded pinning.** Closest to a drop-in fix within the PyTorch ecosystem. Would need evaluation of how it integrates with our custom dataset/collation.
4. **CUDA-stream prefetcher.** Overlaps h2d with compute. May already be partially achieved by `non_blocking=True`, but a dedicated prefetcher with its own stream could help more.
5. **SPDL or FFCV.** Larger integration effort, but they address the root cause (process-based IPC + single-threaded pinning) rather than working around it.

## Other relevant PyTorch issues

- [Very high CPU utilization with pin_memory=True and num_workers > 0](https://github.com/pytorch/pytorch/issues/25010) — all cores pinned at 100%
- [DataLoader can get stuck inside pin_memory](https://github.com/pytorch/pytorch/issues/24927) — hangs after hours of training
- [PyTorch rounds up pinned allocations to powers of 2](https://github.com/pytorch/pytorch/issues/150517) — a 4.5 GB request becomes 8 GB
- [Allocating pinned memory uses twice as much memory](https://github.com/pytorch/pytorch/issues/95823) — for allocations slightly over 128 MB
- [Memory leak from pin_memory_loop](https://github.com/pytorch/pytorch/issues/97432) — `cudaHostAlloc` calls without corresponding frees
- [Pin memory in subprocess](https://github.com/pytorch/pytorch/issues/130124) — user tried pinning in custom subprocesses, hit CUDA OOM
