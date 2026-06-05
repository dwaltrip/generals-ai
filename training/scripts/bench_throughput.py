"""Modal throughput benchmarks for the BC model — two independent probes.

Motivation. A real training run's samples/sec is `min(A, B)` of two separate
ceilings that get conflated when you only watch the training log:

  A. **GPU compute ceiling** — how fast the card can do fwd+bwd for a given
     (width, GPU, batch size, precision), with data loading removed entirely.
  B. **Data-pipeline ceiling** — how fast the `IterableDataset` + `DataLoader`
     can *produce* batches (obs-build is the cost), as a function of cores and
     `num_workers`, with the GPU removed entirely.

If you measure end-to-end with too few workers, A is hidden behind B and a
width sweep ends up measuring the data pipeline instead of the model (every
width pinned at the same worker-bound rate). These two probes measure A and B
in isolation so the real-run rate — and the worker count needed to feed a given
card — both fall out.

Two entrypoints, selected with `::`:

  # A — GPU compute ceiling. Synthetic on-device batches, no Volume, runs the
  #     exact train_one_epoch step. Cells fan out in PARALLEL (independent cards,
  #     no shared resource), OOM-capped per card.
  uv run modal run training/scripts/bench_throughput.py::gpu_ceiling \\
      --gpus T4,L4,A100,H100 --batch-sizes 512,1024,2048,4096

  # B — data-pipeline ceiling. Real DataLoader over a Volume slice, no model.
  #     Cells run SEQUENTIALLY (shared Volume — parallel cells would contend on
  #     I/O and corrupt the numbers). Threads pinned to 1/worker so parallelism
  #     comes only from worker count, the variable under study.
  uv run modal run training/scripts/bench_throughput.py::dataloader \\
      --cpus 4,8,16,32 --workers 1,2,4,8,16,32

Both write a JSON record under `training/data/bench/` and print a table.

Caveats baked into the numbers, surfaced in the output:
  - Precision is fp16 AMP (what `precision=auto` resolves to on CUDA), matching
    real runs. MFU is reported against the **fp32** peak table in `shared.perf`
    (the only one we have sourced); since the work runs on fp16 tensor cores,
    true tensor-core utilization is a good deal lower than the printed MFU.
  - `cudnn.benchmark=True` is set here (correct for our fixed 32x32 shapes) but
    `train.py` does NOT set it today — so realizing this ceiling means enabling
    it there too. Flagged in the output.
  - Probe B measures batch *production* rate (CPU-only, `pin_memory=False`); the
    pinned-copy + H2D transfer overlap with GPU compute in the real loop and are
    out of scope here.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINING_REQS = REPO_ROOT / "training" / "modal_requirements.txt"

# Pin BLAS/OMP thread pools to 1 at the image level (before numpy/torch import).
# Probe B's whole premise is "parallelism == num_workers"; if each worker's
# numpy spun up cores-many BLAS threads, 16 workers x 16 threads would thrash a
# 16-core box and the throughput-vs-workers curve would be noise. Probe A is
# unaffected (its work is on the GPU).
image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_pip_install(requirements=[str(TRAINING_REQS)])
    .env({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    .add_local_python_source("bc", "shared", "utils", "game_types")
)

app = modal.App("bc-bench", image=image)
parsed_replays_vol = modal.Volume.from_name("generals-ai.parsed-replays")

# width set — 0.5x PoC width -> 1x "full-width" (from DeepNash).
# The 3 meaningful widths from the 5.31-3 sweep, so ceiling numbers
# line up with the loss-curve runs.
DEFAULT_WIDTHS = "0.5x:128:128:160,0.75x:192:192:240,1x:256:256:320"


# ======================================================================
# Probe A — GPU compute ceiling
# ======================================================================


def _synth_batch(B: int, cfg, device):
    """A random on-device batch shaped like `dataset.encode_frame`'s collation.

    All-True masks = worst-case (every cell legal, full board) so the heads do
    maximum work; `is_pass` all-False so the policy CE actually runs over the
    whole batch (the realistic heavy path). Values are random — we time the
    step, we don't check the loss.
    """
    import torch

    in_ch, H, W = cfg.in_ch, cfg.H, cfg.W
    return {
        "obs": torch.randn(B, in_ch, H, W, device=device),
        "valid_mask": torch.ones(B, 1, H, W, dtype=torch.bool, device=device),
        "mask": torch.ones(B, H, W, 8, dtype=torch.bool, device=device),
        "action_target": torch.randint(0, H * W * 8, (B,), dtype=torch.int64, device=device),
        "is_pass": torch.zeros(B, dtype=torch.bool, device=device),
        "value_target": torch.randint(0, 8, (B,), dtype=torch.int64, device=device),
    }


def _flops_per_sample(model, device) -> int:
    """Total fwd+bwd FLOPs at B=1 — structural, so precision-independent.

    Surrogate loss = sum across all three heads, matching the *structure* (not
    the value) of `bc_loss` so the backward graph touches every path. Mirrors
    `flops_probe.py` and `train._measure_model_flops_per_sample`.
    """
    import torch

    from shared.perf import measure_total_flops

    def fwd_bwd() -> torch.Tensor:
        x = torch.zeros(1, model.cfg.in_ch, model.cfg.H, model.cfg.W, device=device)
        vm = torch.ones(1, 1, model.cfg.H, model.cfg.W, dtype=torch.bool, device=device)
        out = model(x, vm)
        return out["policy_logits"].sum() + out["pass_logit"].sum() + out["value_logits"].sum()

    flops = measure_total_flops(fwd_bwd)
    model.zero_grad()
    return flops


@app.function(gpu="T4", timeout=60 * 30)  # gpu overridden per-call via with_options
def bench_gpu_ceiling(
    gpu_label: str,
    widths: list[tuple[str, int, int, int]],
    batch_sizes: list[int],
    warmup: int,
    iters: int,
    min_seconds: float,
    precision: str,
) -> dict:
    """Time the real train step over width x batch_size on one card, no data.

    For each cell: build the model, synthesize an on-device batch, run the
    exact `train_one_epoch` step (autocast + GradScaler + optim.step), warm up
    (covers cuDNN autotune + allocator), then time a window of `iters` steps OR
    `min_seconds` wall-clock (whichever is longer) inside a single
    `cuda.synchronize()`-bracketed region. The time floor keeps the tiny/fast
    widths — where a fixed `iters` would be a ~100ms window — statistically
    stable, which matters because comparing widths is the whole point.

    This is the *realistic* ceiling, not a theoretical fully-pipelined one: the
    step calls `bc_loss`, which does a per-step host sync (`n_non_pass.item()`),
    exactly as the real loop does. So the number includes that intrinsic sync
    rather than pretending steps pipeline with zero host involvement.

    Batch sizes ascend; the first OOM for a width breaks the inner loop (larger
    batches would only OOM too) and records a DNF cell.
    """
    import gc
    import time

    import torch

    from bc.loss import bc_loss
    from bc.model import BCModel
    from bc.model_config import build_model_cfg
    from shared.perf import compute_mfu, peak_tflops_fp32

    # Fixed input shapes -> autotune the best kernels once, reuse them. Correct
    # for this workload; note that train.py does NOT set this today.
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    cuda_name = torch.cuda.get_device_name(0)
    peak_fp32 = peak_tflops_fp32(device)
    amp_dtype = torch.float16 if precision == "fp16" else None

    def train_step(model, optim, scaler, batch) -> None:
        optim.zero_grad()
        with torch.amp.autocast(
            "cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None,
        ):
            out = model(batch["obs"], batch["valid_mask"])
            losses = bc_loss(out, batch)
        scaler.scale(losses["total"]).backward()
        scaler.step(optim)
        scaler.update()

    rows = []
    for label, o, m, i in widths:
        cfg = build_model_cfg(
            outer_width=o, middle_width=m, inner_width=i, value_head_variant="pyramid",
        )
        # FLOPs/sample once per width (B-independent), then a clean teardown so
        # the timed model starts from an empty allocator.
        fmodel = BCModel(cfg).to(device).train()
        n_params = sum(p.numel() for p in fmodel.parameters())
        flops_per_sample = _flops_per_sample(fmodel, device)
        del fmodel
        gc.collect()
        torch.cuda.empty_cache()

        for B in batch_sizes:
            model = optim = scaler = batch = None
            try:
                model = BCModel(cfg).to(device).train()
                optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
                scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is not None)
                batch = _synth_batch(B, cfg, device)

                for _ in range(warmup):
                    train_step(model, optim, scaler, batch)
                torch.cuda.synchronize()

                t0 = time.perf_counter()
                n_steps = 0
                while n_steps < iters or (time.perf_counter() - t0) < min_seconds:
                    train_step(model, optim, scaler, batch)
                    n_steps += 1
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                sps = n_steps * B / elapsed
                achieved_tflops = sps * flops_per_sample / 1e12
                rows.append({
                    "width": label, "outer_width": o, "batch_size": B,
                    "params": n_params, "flops_per_sample": flops_per_sample,
                    "samples_per_sec": round(sps, 1),
                    "step_ms": round(elapsed / n_steps * 1e3, 2),
                    "n_steps": n_steps,
                    "achieved_tflops": round(achieved_tflops, 2),
                    "mfu_fp32": (round(compute_mfu(sps, flops_per_sample, peak_fp32), 4)
                                 if peak_fp32 is not None else None),
                    "oom": False,
                })
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                    raise
                rows.append({"width": label, "outer_width": o, "batch_size": B,
                             "params": n_params, "oom": True})
                break  # ascending batch sizes — larger will OOM too
            finally:
                del model, optim, scaler, batch
                gc.collect()
                torch.cuda.empty_cache()

    return {"gpu": gpu_label, "cuda_name": cuda_name, "peak_fp32": peak_fp32, "rows": rows}


# ======================================================================
# Probe B — data-pipeline ceiling
# ======================================================================


def _pin_worker_threads(worker_id: int) -> None:
    """DataLoader worker_init_fn: belt-and-suspenders thread pin inside each
    worker (the image env already sets OMP/BLAS=1; this also pins torch)."""
    import torch

    torch.set_num_threads(1)


# memory headroom: the largest cell (32 workers) holds prefetch_factor(2) x 32
# collated obs batches in flight (~196 MiB each at bs=512) plus 32 per-worker
# shuffle buffers — ~15+ GiB. Request generously so a tight host can't silently
# page/throttle that cell and depress its rate.
@app.function(volumes={"/data": parsed_replays_vol}, memory=32768, timeout=60 * 30)
def bench_dataloader(
    cpu_count: float,
    num_workers: int,
    manifest_path: str,
    intermediate_path: str,
    batch_size: int,
    shuffle_buffer: int,
    warmup_seconds: float,
    timed_seconds: float,
    min_batches: int,
) -> dict:
    """Measure batch *production* rate of the real loader, no model, no GPU.

    Builds the production `IterableDataset` over the manifest's train split and
    iterates dropping each batch, timing a steady-state window after a warmup
    that covers worker spin-up + the per-worker shuffle-buffer fill. `cpu_count`
    is the cores this container was provisioned (set by the caller via
    `with_options(cpu=...)`) — passed in only for labeling.

    The timed window runs for `timed_seconds` OR until `min_batches` have been
    counted, whichever is longer. The time bound keeps fast cells from running
    long; the batch floor keeps slow (1-2 worker) cells from being decided by
    only a handful of high-variance games. Warmup is purely time-bounded.

    NOTE: a short window only ever reads *cold* Volume files (each game once,
    nothing re-read), so this measures ~epoch-1 throughput, not the warm-cache
    steady state a multi-epoch run reaches. Conservative by design.

    `shuffle_buffer` is kept modest by default: in steady state the reservoir
    yields one item per upstream pull, so its size doesn't change the production
    rate (only warmup latency + per-worker memory) — but a 2048 buffer x 32
    workers would blow memory for no throughput signal.
    """
    import time

    import torch
    from torch.utils.data import DataLoader

    from bc.dataset import IterableDataset
    from bc.model_config import MODEL_CONFIG_DEFAULTS
    from bc.splits import load_manifest, samples_for_split

    torch.set_num_threads(1)

    manifest = load_manifest(Path(manifest_path))
    train_samples = samples_for_split(manifest, "train", Path(intermediate_path))

    ds = IterableDataset(
        samples=train_samples, seed=0,
        obs_cfg=MODEL_CONFIG_DEFAULTS.obs, shuffle_buffer_size=shuffle_buffer,
    )
    ds.set_epoch(0)

    dl_kwargs: dict = {
        "batch_size": batch_size, "num_workers": num_workers,
        "pin_memory": False, "worker_init_fn": _pin_worker_threads,
    }
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2
    loader = DataLoader(ds, **dl_kwargs)

    it = iter(loader)
    n = 0
    n_batches = 0
    t0 = time.perf_counter()
    exhausted = False
    try:
        warm_end = t0 + warmup_seconds
        while time.perf_counter() < warm_end:
            next(it)
        # Reset the timer after warmup (worker spin-up + buffer fill excluded).
        n = 0
        n_batches = 0
        t0 = time.perf_counter()
        timed_end = t0 + timed_seconds
        while time.perf_counter() < timed_end or n_batches < min_batches:
            n += next(it)["obs"].shape[0]
            n_batches += 1
        elapsed = time.perf_counter() - t0
    except StopIteration:
        # Slice smaller than the window — shouldn't happen on cloud_24k, but
        # report what we got rather than crash.
        exhausted = True
        elapsed = time.perf_counter() - t0

    sps = n / elapsed if elapsed > 0 else 0.0
    return {
        "cpu": cpu_count, "num_workers": num_workers, "batch_size": batch_size,
        "samples_per_sec": round(sps, 1), "n_samples": n, "n_batches": n_batches,
        "elapsed_sec": round(elapsed, 2),
        "per_worker_sps": round(sps / num_workers, 1) if num_workers > 0 else round(sps, 1),
        "exhausted": exhausted,
    }


# ======================================================================
# Local entrypoints (run on the laptop; fan out + aggregate + report)
# ======================================================================


def _parse_widths(s: str) -> list[tuple[str, int, int, int]]:
    out = []
    for cell in s.split(","):
        label, o, m, i = cell.split(":")
        out.append((label, int(o), int(m), int(i)))
    return out


def _write_results(kind: str, payload: dict) -> Path:
    out_dir = REPO_ROOT / "training" / "data" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = out_dir / f"{kind}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


@app.local_entrypoint()
def gpu_ceiling(
    gpus: str = "T4,L4,A100,H100",
    widths: str = DEFAULT_WIDTHS,
    batch_sizes: str = "512,1024,2048,4096",
    warmup: int = 15,
    iters: int = 50,
    min_seconds: float = 2.0,
    precision: str = "fp16",
) -> None:
    """Probe A: fan the GPU-ceiling bench across cards in parallel, then report."""
    gpu_list = [g.strip() for g in gpus.split(",") if g.strip()]
    width_list = _parse_widths(widths)
    bs_list = [int(b) for b in batch_sizes.split(",") if b.strip()]

    print(f"GPU compute ceiling — {precision} | widths={[w[0] for w in width_list]} | "
          f"batch_sizes={bs_list} | warmup={warmup} iters={iters} (>= {min_seconds}s)")
    print(f"cards (parallel): {gpu_list}\n")

    # Synthetic data -> no shared resource -> safe to run cards concurrently.
    handles = [(g, bench_gpu_ceiling.with_options(gpu=g).spawn(
        g, width_list, bs_list, warmup, iters, min_seconds, precision)) for g in gpu_list]
    results = [h.get() for _, h in handles]

    for r in results:
        peak = r["peak_fp32"]
        peak_str = f"{peak} TFLOPS fp32-peak" if peak is not None else "peak unknown"
        print(f"=== {r['gpu']}  ({r['cuda_name']}, {peak_str}) ===")
        print(f"  {'width':>6} {'bs':>5} {'sps':>9} {'step_ms':>8} {'TFLOPS':>8} {'MFU(fp32)':>10}")
        for row in r["rows"]:
            if row.get("oom"):
                print(f"  {row['width']:>6} {row['batch_size']:>5} {'OOM':>9}")
                continue
            mfu = f"{row['mfu_fp32'] * 100:.1f}%" if row["mfu_fp32"] is not None else "n/a"
            print(f"  {row['width']:>6} {row['batch_size']:>5} {row['samples_per_sec']:>9,.0f} "
                  f"{row['step_ms']:>8} {row['achieved_tflops']:>8} {mfu:>10}")
        print()

    path = _write_results("gpu_ceiling", {
        "kind": "gpu_ceiling", "precision": precision,
        "warmup": warmup, "iters": iters, "min_seconds": min_seconds, "results": results,
    })
    print(f"wrote {path}")
    print("NOTE: cudnn.benchmark=True here but NOT in train.py — enable it there to realize these.")


@app.local_entrypoint()
def dataloader(
    scaling_cpu: float = 32.0,
    scaling_workers: str = "1,2,4,8,16,32",
    contention_cpus: str = "4,8,16",
    manifest_path: str = "/data/cloud_24k.json",
    intermediate_path: str = "/data/intermediate-cloud-20k",
    batch_size: int = 512,
    shuffle_buffer: int = 256,
    warmup_seconds: float = 15.0,
    timed_seconds: float = 25.0,
    min_batches: int = 20,
) -> None:
    """Probe B: lean grid, run SEQUENTIALLY (shared Volume — concurrent cells
    would contend on Volume I/O and corrupt the rates).

    Two parts, because the `cpu=` axis only bites once `num_workers` approaches
    the core count:
      - **Clean scaling curve** at a fixed high `scaling_cpu` (cores never the
        bottleneck): isolates per-worker rate, linearity, and the plateau where
        the single-threaded main-process collate — not cores — caps production.
      - **Contention probe**: at each `contention_cpus` value, `workers = cpu`
        and `workers = 2*cpu`, to find the oversubscription penalty.
    This drops the redundant low-worker cells a full cpu×worker cross-product
    repeats (a 1-worker cell is ~identical at cpu=4 or cpu=32).
    """
    # `if t.strip()` so an empty flag (e.g. --contention-cpus "") yields an
    # empty list instead of choking on a "" token.
    sworkers = [int(w) for w in scaling_workers.split(",") if w.strip()]
    ccpus = [float(c) for c in contention_cpus.split(",") if c.strip()]

    print(f"Data-pipeline ceiling (lean grid) — manifest={manifest_path} bs={batch_size} "
          f"buffer={shuffle_buffer}")
    print(f"  warmup={warmup_seconds}s  timed={timed_seconds}s (>= {min_batches} batches)\n")

    rows = []
    seen: set[tuple[float, int]] = set()

    def run_cell(cpu: float, w: int) -> None:
        if (cpu, w) in seen:
            return
        seen.add((cpu, w))
        # cpu=(request, limit) HARD-CAPS the container to `cpu` cores. A scalar
        # `cpu=` is only a *request* — the container could burst to all host
        # cores, so a workers=2*cpu contention cell would never actually
        # oversubscribe and the penalty would vanish. The tuple is load-bearing;
        # do not collapse it to a scalar.
        r = bench_dataloader.with_options(cpu=(cpu, cpu)).remote(
            cpu, w, manifest_path, intermediate_path,
            batch_size, shuffle_buffer, warmup_seconds, timed_seconds, min_batches)
        rows.append(r)
        print(f"  cpu={cpu:>4} workers={w:>3} -> {r['samples_per_sec']:>8,.0f} sps "
              f"({r['per_worker_sps']:,.0f}/worker, {r['n_batches']} batches)"
              + ("  [slice exhausted]" if r["exhausted"] else ""))

    print(f"clean scaling curve @ cpu={scaling_cpu}:")
    for w in sworkers:
        run_cell(scaling_cpu, w)

    print("contention probe (workers = cpu and 2x cpu):")
    for c in ccpus:
        run_cell(c, int(c))
        run_cell(c, int(2 * c))

    path = _write_results("dataloader", {
        "kind": "dataloader", "manifest_path": manifest_path,
        "batch_size": batch_size, "shuffle_buffer": shuffle_buffer,
        "warmup_seconds": warmup_seconds, "timed_seconds": timed_seconds,
        "min_batches": min_batches, "results": rows,
    })
    print(f"\nwrote {path}")
