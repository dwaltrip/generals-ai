"""BC training runner.

Takes a validated `TrainConfig`, drives an end-to-end training run.
Pure runner: no CLI, no environment-specific defaults. CLI scaffolding
lives in `bc.train_cli`; the config dataclass + run-id factory live in
`bc.train_config`; entry-point wrappers in `training/scripts/`.

Reads a split manifest produced by `bc.splits build`, walks the train split
via `IterableDataset`, runs N epochs of AdamW SGD with `bc_loss`. After each
epoch, runs a full validation pass via `bc.eval.run_val` and saves a
checkpoint. Prints per-batch component losses + rolling samples/sec every
`log_every` batches; end-of-epoch summary collects the sample-weighted
means via `LossAccumulator`.

Writes into `config.run_dir`, which must already exist — callers
initialize it via `initialize_run_dir(config)` before invoking `bc_run`.
This lets cloud entry points drop sibling provenance files (e.g.
`args_cloud.json`) into the run dir *before* training starts.

Files produced:
  - `args.json`         — full config as JSON, for provenance.
  - `batches.jsonl`     — one record per batch.
  - `epochs.jsonl`      — one record per epoch (train + val summary, ckpt name).
  - `checkpoints/epoch_NNN.pt` — combined dict (model + optim + scaler + epoch) per epoch.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bc.checkpoint import ckpt_name
from bc.constants import H_PADDED, OBS_CHANNELS, W_PADDED
from bc.dataset import IterableDataset, assert_safe_loader
from bc.eval import run_val
from bc.loss import LossAccumulator, bc_loss
from bc.model import BCModel
from bc.run_dir import RunArtifacts
from bc.run_logger import RunLogger
from bc.splits import load_manifest, samples_for_split
from bc.state import TrainingState
from bc.train_config import TrainConfig
from shared.device import (
    dataloader_kwargs,
    disable_mps_fallback,
    move_batch,
    pick_device,
    resolve_precision,
)
from shared.gpu_sidecar import gpu_util_sidecar
from shared.perf import compute_mfu, measure_total_flops, peak_tflops_fp32
from utils.log import tee_stdio


def _fmt_metric(x: float | None, prec: int = 4) -> str:
    """Format a metric for console output, with `n/a` fallback for `None`.
    `run_val` returns `None` for accuracies whose denominators are 0."""
    return f"{x:.{prec}f}" if x is not None else "n/a"


def _measure_model_flops_per_sample(model: BCModel, device: torch.device) -> int:
    """Run one synthetic forward+backward at B=1, return total counted FLOPs.

    Used at training start to derive a per-run FLOPs/sample constant for
    MFU reporting. Cost is one extra fwd+bwd at startup — negligible
    relative to a real epoch.

    Zeros grads afterward so the measurement doesn't contaminate the
    first real optimizer step. Restores the model's train/eval flag.
    """
    was_training = model.training
    model.train()

    def fwd_bwd() -> torch.Tensor:
        x = torch.zeros(1, OBS_CHANNELS, H_PADDED, W_PADDED, device=device)
        # All-True mask: synthetic-throughput measurement assumes a fully-used
        # grid (worst-case FLOPs through the heads).
        valid_mask = torch.ones(1, 1, H_PADDED, W_PADDED, dtype=torch.bool, device=device)
        out = model(x, valid_mask)
        # Sum across all three heads so the backward graph touches every
        # path — matches the structure (not the value) of `bc_loss`.
        return out["policy_logits"].sum() + out["pass_logit"].sum() + out["value_logits"].sum()

    flops = measure_total_flops(fwd_bwd)
    model.zero_grad()
    if not was_training:
        model.eval()
    return flops


def train_one_epoch(
    epoch: int,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    dataset: IterableDataset,
    loader: DataLoader,
    device: torch.device,
    logger: RunLogger,
    run_start: float,
    max_batches: int | None,
    log_every: int,
    scaler: torch.amp.GradScaler,
    amp_dtype: torch.dtype | None,
) -> dict:
    """Run one epoch of BC training.

    Iterates `loader`, performs forward/backward/optim.step per batch,
    writes per-batch JSONL records via `logger`, and prints a console
    line every `log_every` batches with a rolling samples/sec rate.

    Returns the epoch summary dict: sample-weighted mean losses (via
    `LossAccumulator`) plus `n_batches`, `duration_sec`, `samples_per_sec`.
    Caller owns the per-epoch JSONL row and ckpt save.
    """
    # Reset train mode every epoch — pairs with `run_val`'s `model.eval()`
    # so each epoch's train pass starts in train mode regardless of where
    # the previous epoch left the model.
    model.train()

    # Advance the dataset's per-epoch shuffle. Mutating an internal
    # counter from `IterableDataset.__iter__` wouldn't survive the
    # DataLoader worker fork on `num_workers > 0`; the caller-side
    # `set_epoch` is the supported entry point. We hold a typed `dataset`
    # reference rather than going through `loader.dataset` because the
    # latter is typed as the base `Dataset` (no `set_epoch`).
    dataset.set_epoch(epoch)

    acc = LossAccumulator()
    epoch_start = time.perf_counter()
    n_batches_seen = 0

    # Rolling samples/sec — instantaneous rate across the last
    # log-every window. Reset every print so the number tracks
    # current throughput rather than smoothing over the epoch.
    window_start = epoch_start
    window_samples = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_batch(batch, device)
        optim.zero_grad()
        # AMP path: autocast promotes the matmul/conv-heavy forward to
        # fp16 (tensor-core eligible on CUDA); numerically-sensitive
        # ops like GroupNorm + softmax stay in fp32 per autocast's
        # built-in op-routing rules. `scaler` is a GradScaler that is
        # disabled when `amp_dtype is None` — `scale()`/`step()`/`update()`
        # become near-no-ops, so the fp32 path stays unchanged.
        with torch.amp.autocast(
            device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None,
        ):
            out = model(batch["obs"], batch["valid_mask"])
            losses = bc_loss(out, batch)
        scaler.scale(losses["total"]).backward()
        scaler.step(optim)
        scaler.update()

        B = batch["obs"].shape[0]
        acc.update(losses, batch_size=B)
        window_samples += B
        n_batches_seen += 1

        logger.log_batch({
            "epoch": epoch,
            "batch_idx": batch_idx,
            "batch_size": B,
            "policy": float(losses["policy"].item()),
            "value": float(losses["value"].item()),
            "pass": float(losses["pass"].item()),
            "total": float(losses["total"].item()),
            "n_non_pass": int(losses["n_non_pass"].item()),
            "wall_time_sec": round(time.perf_counter() - run_start, 3),
        })

        if (batch_idx + 1) % log_every == 0:
            rate = window_samples / (time.perf_counter() - window_start)
            print(
                f"[epoch {epoch}] batch {batch_idx + 1} | "
                f"policy {losses['policy'].item():6.4f} "
                f"value {losses['value'].item():6.4f} "
                f"pass {losses['pass'].item():6.4f} "
                f"total {losses['total'].item():6.4f} | "
                f"{rate:.0f} samples/sec"
            )
            window_start = time.perf_counter()
            window_samples = 0

    epoch_dur = time.perf_counter() - epoch_start
    s = acc.summary()
    rate = s["n_samples"] / epoch_dur if epoch_dur > 0 else 0.0
    return {
        **s,
        "n_batches": n_batches_seen,
        "duration_sec": round(epoch_dur, 3),
        "samples_per_sec": round(rate, 2),
    }


def build_dataloader(
    config: TrainConfig,
    train_samples: list[tuple[Path, int]],
    device: torch.device,
) -> tuple[IterableDataset, DataLoader]:
    """Build the train `IterableDataset` + `DataLoader` from a config.

    Returns both: the typed dataset (so the loop can call `set_epoch`,
    which `loader.dataset` doesn't expose) and the loader. Prints the
    resolved dataloader settings — `pin_memory`/`prefetch_factor` are
    device-dependent, so the log shows resolved value vs. config request.
    """
    ds = IterableDataset(
        samples=train_samples,
        seed=config.seed,
        shuffle_buffer_size=config.shuffle_buffer_size,
    )
    dl_kwargs = dataloader_kwargs(
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        device=device,
    )
    loader = DataLoader(ds, batch_size=config.batch_size, **dl_kwargs)
    assert_safe_loader(loader)
    print(
        f"dataloader: batch_size={config.batch_size}  "
        f"num_workers={dl_kwargs['num_workers']}  "
        f"pin_memory={dl_kwargs['pin_memory']} (config={config.pin_memory!r})  "
        f"prefetch_factor={dl_kwargs.get('prefetch_factor', 'n/a')}"
    )
    return ds, loader


def bc_run(config: TrainConfig) -> None:
    """Drive a BC training run end-to-end from a validated config.

    Precondition: `config.run_dir` exists. Loads the manifest and builds
    the train dataset, then hands off to `run_loop` for model build +
    training. Callable from a notebook or test by constructing a
    `TrainConfig` directly (after initializing the run dir).
    """
    disable_mps_fallback()

    if not config.manifest.exists():
        raise SystemExit(f"manifest not found: {config.manifest}")
    if not config.intermediate.exists():
        raise SystemExit(f"intermediate corpus not found: {config.intermediate}")
    if not config.run_dir.exists():
        raise SystemExit(f"run_dir not found: {config.run_dir}")

    device = pick_device(config.device)
    torch.manual_seed(config.seed)

    ckpt_dir = config.run_dir / "checkpoints"
    ckpt_dir.mkdir()

    # Tee from here so everything past the run-dir setup — manifest load,
    # dataset summary, model build, training — lands in run.log. The
    # "run dir:" line printed by `initialize_run_dir` stays terminal-only;
    # self-reference inside the log would just be noise.
    with tee_stdio(config.run_dir / "run.log"):
        # --- Manifest + dataset ---
        print(f"loading manifest: {config.manifest}")
        manifest = load_manifest(config.manifest)
        train_samples = samples_for_split(manifest, "train", config.intermediate)
        val_samples = samples_for_split(manifest, "val", config.intermediate)
        print(
            f"  filter_version={manifest['filter_version']}  "
            f"git_sha={manifest['git_sha']}  "
            f"kept_pairs={manifest['kept_pairs']:,}  "
            f"train_pairs={len(train_samples):,}  "
            f"val_pairs={len(val_samples):,}"
        )

        ds, loader = build_dataloader(config, train_samples, device)

        run_loop(
            config=config,
            device=device,
            dataset=ds,
            loader=loader,
            val_samples=val_samples,
            run_dir=config.run_dir,
            ckpt_dir=ckpt_dir,
        )


def print_epoch_summary(epoch: int, summary: dict, val_summary: dict | None) -> None:
    """Print the end-of-epoch console block: train means + throughput,
    then the val block (or a skip notice). Reads MFU from `summary["mfu"]`
    (already computed by the caller); omits the MFU clause when it's `None`
    (unknown device peak)."""
    print()
    mfu_str = (
        f" | MFU {summary['mfu'] * 100:.1f}%"
        if summary["mfu"] is not None else ""
    )
    print(
        f"[epoch {epoch}] complete | "
        f"{summary['n_samples']:,} frames ({summary['n_non_pass']:,} non-pass) "
        f"in {summary['duration_sec']:.1f}s | "
        f"{summary['samples_per_sec']:.0f} samples/sec"
        f"{mfu_str} ({summary['n_batches']} batches)"
    )
    print(
        f"[epoch {epoch}] mean: "
        f"policy {summary['policy']:.4f}  "
        f"value {summary['value']:.4f}  "
        f"pass {summary['pass']:.4f}  |  "
        f"total {summary['total']:.4f}"
    )
    if val_summary is not None:
        print(
            f"[epoch {epoch}] val | "
            f"{val_summary['n_samples']:,} frames ({val_summary['n_non_pass']:,} non-pass) "
            f"in {val_summary['duration_sec']:.1f}s | "
            f"{val_summary['samples_per_sec']:.0f} samples/sec | "
            f"policy {val_summary['policy']:.4f}  "
            f"value {val_summary['value']:.4f}  "
            f"pass {val_summary['pass']:.4f}  |  "
            f"total {val_summary['total']:.4f}"
        )
        print(
            f"[epoch {epoch}] val | "
            f"top1 {_fmt_metric(val_summary['top1'])}  "
            f"top3 {_fmt_metric(val_summary['top3'])}  "
            f"pass_acc {_fmt_metric(val_summary['pass_acc'])}  "
            f"pass_frac {_fmt_metric(val_summary['pass_frac'])}"
        )
    else:
        print(f"[epoch {epoch}] val skipped (--skip-val)")


def run_loop(
    config: TrainConfig,
    device: torch.device,
    dataset: IterableDataset,
    loader: DataLoader,
    val_samples: list[tuple[Path, int]],
    run_dir: Path,
    ckpt_dir: Path,
) -> None:
    """Build the model + optimizer and drive the epoch loop.

    Per epoch: train pass via `train_one_epoch`, full val pass via
    `run_val`, checkpoint save, one row to `epochs.jsonl`, console
    summary. JSONL handles (owned by `RunArtifacts`) are line-buffered and
    closed on context exit so `tail -f` works and a mid-epoch raise still
    flushes records to disk.
    """
    # --- Model + optimizer ---
    print(f"building model on {device} (value_head={config.value_head_variant})")
    state = TrainingState.fresh(config, device)
    n_params = sum(p.numel() for p in state.model.parameters())
    print(f"  params: {n_params:,}")

    # Mixed precision resolution. `amp_dtype is None` is the canonical
    # "AMP off" sentinel — autocast is wired through both code paths so the
    # fp32 case is a no-op rather than a separate branch. The matching
    # GradScaler lives on `state` (built from the same precision decision).
    resolved_precision = resolve_precision(config.precision, device)
    amp_dtype = torch.float16 if resolved_precision == "fp16" else None
    print(f"  precision: {resolved_precision} (config={config.precision!r})")

    # FLOPs/sample + device peak: drives MFU per-epoch. FLOPs are measured
    # once on a synthetic batch; peak comes from a hardcoded device table
    # (`shared.perf`) and is `None` on unknown hardware — in which case
    # MFU is omitted rather than computed against a wrong denominator.
    flops_per_sample = _measure_model_flops_per_sample(state.model, device)
    peak_tflops = peak_tflops_fp32(device)
    print(f"  fwd+bwd FLOPs/sample: {flops_per_sample / 1e9:.2f} GFLOPs")
    if peak_tflops is not None:
        print(f"  device peak FP32:     {peak_tflops:.1f} TFLOPS")
    else:
        print("  device peak FP32:     unknown (MFU will be omitted)")
    print()

    # --- Train loop ---
    run_start = time.perf_counter()
    artifacts = RunArtifacts(
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        logger=RunLogger(run_dir),
        flops_per_sample=flops_per_sample,
        peak_tflops=peak_tflops,
    )
    # `RunArtifacts` opens/closes the JSONL writers; the gpu sidecar is a
    # sibling context manager. Both unwind on a mid-epoch raise, flushing
    # the line-buffered records to disk.
    with artifacts, gpu_util_sidecar(run_dir, device):
        for epoch in range(1, config.epochs + 1):
            summary = train_one_epoch(
                epoch=epoch,
                model=state.model,
                optim=state.optim,
                dataset=dataset,
                loader=loader,
                device=device,
                logger=artifacts.logger,
                run_start=run_start,
                max_batches=config.max_batches,
                log_every=config.log_every,
                scaler=state.scaler,
                amp_dtype=amp_dtype,
            )
            if config.skip_val:
                val_summary = None
            else:
                val_summary = run_val(
                    model=state.model,
                    val_samples=val_samples,
                    device=device,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    pin_memory=config.pin_memory,
                    prefetch_factor=config.prefetch_factor,
                    seed=config.seed,
                    amp_dtype=amp_dtype,
                )

            # Augment the epoch summary with MFU (None when peak is unknown) and the
            # FLOPs constant used to compute it, so the jsonl row is self-describing.
            mfu = compute_mfu(
                summary["samples_per_sec"], artifacts.flops_per_sample, artifacts.peak_tflops
            )
            summary["mfu"] = round(mfu, 4) if mfu is not None else None
            summary["flops_per_sample"] = artifacts.flops_per_sample

            # Write the epoch row + print the summary BEFORE saving the checkpoint.
            # A checkpoint-save failure shouldn't lose the epoch's computed metrics.
            ckpt_file = ckpt_name(epoch)
            artifacts.logger.log_epoch({
                "epoch": epoch,
                **summary,
                "val": val_summary,
                "ckpt": ckpt_file,
            })
            print_epoch_summary(epoch, summary, val_summary)

            # Commit the epoch: bump `state.epoch` to the just-completed value,
            # then save. The bump happens only after the epoch's work + metrics
            # are persisted, so a crash before here leaves `state.epoch` one
            # behind — exactly what resume expects. Save failure raises out of
            # the loop with metrics already written above.
            state.epoch = epoch
            state.save(artifacts.ckpt_dir)
            print(f"[epoch {epoch}] saved checkpoint: checkpoints/{ckpt_file}")
            print()
