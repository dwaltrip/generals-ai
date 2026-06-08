# GPU performance mental model

Reference for reasoning about training throughput from first principles: how to read a samples-per-second (sps) number, what the realistic ceilings look like, and which knobs actually move them.

Written alongside the cloud spike work in [`2026-05/5.22-2-cloud-real-mvp.md`](./2026-05/5.22-2-cloud-real-mvp.md) — the numbers and gotchas reference our specific BC training setup but the framing generalizes.

## The three bottlenecks

Every training step is gated by whichever of these is currently slowest. Optimization is the loop of (a) identifying which one binds, (b) fixing it, (c) discovering which one now binds:

1. **Compute** — GPU's arithmetic throughput. You're compute-bound when the GPU is busy 100% of the time doing math.
2. **Memory bandwidth** — moving data between HBM (GPU memory) and the compute units. You're memory-bound when the GPU could do more math but is waiting on loads/stores.
3. **Data delivery** — the host pipeline (disk → CPU prep → H2D transfer to GPU). You're data-starved when the GPU is *idle* waiting for the next batch.

Most "my training is slow" mysteries are about figuring out which of the three is currently binding.

## The roofline calculation

The compute floor for a training step:

```
min time per sample = FLOPs per sample / effective TFLOPS of the GPU
```

Worked example with our model on a T4 in FP32:
- Model fwd+bwd: **10.57 GFLOPs/sample** (measured by `packages/training/scripts/flops_probe.py`)
- T4 peak FP32: **8.1 TFLOPS** (vendor-published — always optimistic)
- Theoretical max: 8.1e12 / 10.57e9 = **766 sps**
- Actual measured: **238 sps** (T4 baseline, bs=256 nw=4)
- → 238 / 766 = **~31% MFU**

MFU = Model FLOPs Utilization = "what fraction of the device's theoretical peak compute are we actually using?". The training loop prints this in the per-epoch summary line (see `shared/perf.py`).

### How to read an MFU number

- **40–55%** — frontier-shop transformer training, perfectly tuned.
- **15–35%** — hobby / small-CNN training, normal range. (We sit here.)
- **<10%** on a substantive workload — probably data-starved or kernel-launch-bound, not compute-bound. Worth investigating.

The 69% gap between our 31% and the theoretical peak is *not* one thing — it's a mix of kernel-launch overhead (many small ops), per-layer memory-bandwidth ceilings, optimizer + loss math (not counted in the model's FLOPs), and the host pipeline.

### What you need to compute MFU

Two numbers, both relatively stable:

1. **FLOPs/sample for your specific model.** Param count alone doesn't tell you — a 4M-param transformer over a 1000-token sequence and a 4M-param CNN over a 24×24 board do very different work. Use `torch.utils.flop_counter.FlopCounterMode` (built into PyTorch) on one forward+backward pass.
2. **Peak FLOPS for the device.** Vendor-published, hardcoded in `shared/perf.py` for the devices we use. Different per precision (see "Tensor cores" below).

## Tensor cores: two compute units per GPU

A modern NVIDIA GPU has two kinds of compute units on the same die:

- **CUDA cores** — general-purpose ALUs. The "FP32 TFLOPS" number measures this.
- **Tensor cores** — specialized silicon that does a small matrix multiply (a 4×4 or larger MMA — "matrix multiply-accumulate") in a single hardware instruction. Introduced in Volta (2017). Operates at lower precision (FP16/BF16 input), accumulates internally in FP32.

The "8× faster in FP16" framing isn't about precision itself — it's the gap between general-purpose silicon and specialized silicon, which happen to live behind different precisions.

### Cross-GPU ratios

Rough peak TFLOPS, vendor-published. The *ratios* are the informative part:

| GPU | Year | FP32 (CUDA) | Tensor peak | Ratio |
|---|---|---|---|---|
| V100 | 2017 | 15.7 | 125 (FP16) | ~8× |
| **T4** | 2018 | 8.1 | 65 (FP16) | **~8×** |
| A100 | 2020 | 19.5 | 312 (FP16/BF16) | ~16× |
| H100 | 2022 | 67 | 989 (FP16/BF16) / 1,979 (FP8) | ~15× / ~30× |
| RTX 4090 | 2022 | 83 | 165 (FP16) | ~2× (consumer cap) |

Two patterns:

- **Datacenter GPUs grow the ratio over time.** Newer generations dedicate more silicon to tensor cores. H100's FP8 path is 30× over its own FP32 — that's the kind of ceiling H100 is built to sell.
- **Consumer cards (RTX 30/40 series) have artificially smaller ratios.** NVIDIA caps tensor throughput on gaming cards to push professional users toward datacenter cards. A 4090's AMP win is much smaller than on a "real" datacenter card.

### Apple Silicon

M1/M2/M3/M4 Macs don't have tensor cores in the NVIDIA sense. MPS has FP16 fast paths but the speedup isn't in the same league. Our measured M1 utilization sits at ~20% MFU and doesn't move much under AMP — see [`5.22-2-cloud-real-mvp.md`](./2026-05/5.22-2-cloud-real-mvp.md) for the cal numbers.

## Precision formats

One paragraph each on the precisions you'll encounter:

- **FP32** — 8-bit exponent, 23-bit mantissa. The historical default for training. Slow on tensor cores (they don't accept it).
- **FP16** — 5-bit exponent, 10-bit mantissa. Small range (max ±65,504); gradients can overflow or underflow. Needs `GradScaler` to scale the loss before backward so gradients land in representable range. The original tensor-core precision (Volta/Turing — i.e. T4).
- **BF16** — 8-bit exponent, 7-bit mantissa. Same range as FP32, less precision in the mantissa. **No scaler needed** because no overflow risk. Requires Ampere or newer (A100, H100, RTX 30/40). T4 doesn't have it.
- **TF32** — Ampere+'s "automatic" precision for matmul: same exponent as FP32, mantissa truncated to FP16's 10 bits. Kicks in transparently without code changes, gives ~8× over true FP32 on Ampere. Not on T4.
- **FP8** — H100+ only. The frontier-shop precision for large training runs.

### Practical implication for our setup

- **On T4 (current cloud target):** FP16 + `GradScaler` is our only AMP path. No BF16, no TF32.
- **On H100 (if we ever get there):** BF16 is the no-brainer — simpler ergonomics (no scaler), bigger ceiling.
- **On M1 Max (local dev):** Default to FP32. MPS's autocast is less battle-tested, and we don't see a meaningful win.

## What the knobs actually do

- **Batch size.** Bigger = more parallelism on GPU + amortized per-batch overhead (kernel launches, optimizer step). But: more VRAM, fewer gradient updates per epoch (different learning dynamics, not just speed). Sweet spot exists somewhere between "fully saturating the GPU" and "out of memory."
- **Model size.** Directly scales compute. A larger model on the same hardware moves you *rightward* on the roofline — more FLOPs per byte of memory traffic = higher arithmetic intensity = more likely compute-bound = uses the GPU better. Tiny models tend to leave GPUs underutilized.
- **Precision.** FP32 → FP16/BF16 = the single biggest free win on tensor-core GPUs. Practical wins for small CNNs: 1.5–3×. We measured **+36%** going FP32 → FP16 on our 4.2M-param model on T4.
- **DataLoader workers (`num_workers`).** More workers = more parallel data prep, less GPU idle time waiting for batches. Sweet spot differs per platform: on our M1 cal, nw=2 helped vs nw=0 on real workload; on T4, nw=4 climbed monotonically through what we tested.
- **`max_batches`, `epochs`.** These set *how long the run takes*, not throughput. They don't affect sps.

## AMP footguns we've hit

Two latent bugs in the codebase that only surfaced once AMP was turned on:

1. **`MASK_NEG = -1e9` overflows FP16.** The mask-fill value before softmax was set to -1e9 (was intended to be "large-negative but representable"). FP16 maxes out at ±65,504, so -1e9 overflows and `masked_fill` crashes under autocast. Fix: change to `-1e4` (still well below softmax-underflow threshold, comfortably in FP16's range). See `bc/loss.py`.
2. **Combined fancy indexing on MPS (`topk[non_pass, 0]`).** Bool mask + int column index in one bracket returned garbage indices on MPS. Pre-AMP bug we noticed only when re-running the val pass for the M1 cal. Fix: split into two ops (`topk[:, 0][non_pass]`). See `bc/eval.py`.

The pattern: **first-time AMP rollouts often surface latent bugs in masking, sentinel values, and indexing**. The fixes are small but the failure modes are loud (kernel errors, NaN propagation). Worth budgeting an afternoon when first enabling.

## Operationally — what to check before optimizing

A quick checklist for "is this run as fast as it should be":

1. **Is AMP on?** (CUDA only.) `grep precision args.json` or scan `console.log`. If you're on FP32 on a tensor-core GPU, that's likely the biggest unclaimed win.
2. **What's the current MFU?** Per-epoch summary line. <15% on a non-trivial model → investigate; 30–40% → reasonable.
3. **Is the GPU actually busy?** `gpu_util.jsonl` sidecar (CUDA-only). If `gpu_util_pct` is bouncing between 0 and 100, you're likely data-starved (waiting between batches) rather than compute-bound.
4. **Did you measure FLOPs/sample after model changes?** `./packages/training/scripts/flops_probe.py`. Without this, MFU is wrong.

## Related

- [`working-with-modal-cloud-gpu.md`](./working-with-modal-cloud-gpu.md) — evergreen Modal reference (image setup, lockfile workflow, gotchas).
- [`2026-05/5.22-2-cloud-real-mvp.md`](./2026-05/5.22-2-cloud-real-mvp.md) — cloud spike tracker; carries the per-platform measured numbers this doc references.
- `packages/training/scripts/flops_probe.py` — CLI tool for measuring FLOPs/sample of `BCModel`.
- `packages/training/training/shared/perf.py` — MFU + peak-TFLOPS helpers used by the training runner.
