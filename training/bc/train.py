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

from collections.abc import Callable
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from bc.checkpoint import ckpt_name
from bc.constants import H_PADDED, W_PADDED
from bc.dataset import IterableDataset, assert_safe_loader
from bc.eval import run_val
from bc.loss import LossAccumulator, bc_loss
from bc.model import BCModel
from bc.resume_warmup import WarmupSchedule
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
from shared.resource_info import log_resource_info
from utils.log import abort, tee_stdio


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
        x = torch.zeros(1, model.cfg.in_ch, H_PADDED, W_PADDED, device=device)
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
    warmup: WarmupSchedule | None = None,
) -> dict:
    """Run one epoch of BC training.

    Iterates `loader`, performs forward/backward/optim.step per batch,
    writes per-batch JSONL records via `logger`, and prints a console
    line every `log_every` batches with a rolling samples/sec rate.

    `warmup` (legacy-resume LR ramp) is stepped at the top of each batch when
    present; `None` on fresh runs and combined resumes, where the LR is constant.

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

        # Legacy-resume LR warmup: set this batch's LR on the optimizer before
        # the step. No-op when warmup is None (fresh runs / combined resumes).
        if warmup is not None:
            warmup.step(optim)

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
            "lr": optim.param_groups[0]["lr"],
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
        obs_cfg=config.arch.obs,
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
    log_resource_info(
        device=device,
        batch_size=config.batch_size,
        obs_channels=config.arch.obs.obs_channels,
        spatial=(H_PADDED, W_PADDED),
        num_workers=dl_kwargs["num_workers"],
        prefetch_factor=dl_kwargs.get("prefetch_factor"),
        pin_memory=dl_kwargs["pin_memory"],
    )
    return ds, loader


def bc_run(config: TrainConfig) -> None:
    """Drive a fresh BC training run end-to-end from a validated config.

    The fresh entry point: builds a new `TrainingState` and runs all
    configured epochs. Resume is the sibling entry (`bc.resume.bc_resume`);
    both share `run_training`. Precondition: `config.run_dir` exists. Callable
    from a notebook or test by constructing a `TrainConfig` directly (after
    initializing the run dir).
    """
    run_training(config, suffix="", make_state=lambda dev: TrainingState.fresh(config, dev))


def run_training(
    config: TrainConfig,
    suffix: str,
    make_state: Callable[[torch.device], TrainingState],
) -> None:
    """Set up a training segment and drive its epoch loop.

    Resume-agnostic core shared by the fresh (`bc_run`) and resume
    (`bc.resume.bc_resume`) entry points. `suffix` names the segment's
    artifact files ("" for the original run, "_resume_NN" for resumes);
    `make_state` builds the `TrainingState` (`TrainingState.fresh` for a fresh
    run, `TrainingState.from_checkpoint` for a resume), so this function never
    needs to know which it is.

    Loads the manifest + dataset, builds the model and run-start FLOPs
    measurements, then hands the epoch loop to `train_loop`. Tees stdout +
    stderr to `run{suffix}.log` for the duration.
    """
    disable_mps_fallback()

    if not config.manifest.exists():
        abort(f"manifest not found: {config.manifest}")
    if not config.intermediate.exists():
        abort(f"intermediate corpus not found: {config.intermediate}")
    if not config.run_dir.exists():
        abort(f"run_dir not found: {config.run_dir}")

    device = pick_device(config.device)
    torch.manual_seed(config.seed)

    ckpt_dir = config.run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Tee from here so everything past the run-dir setup — manifest load,
    # dataset summary, model build, training — lands in run{suffix}.log. The
    # "run dir:" line printed by `initialize_run_dir` stays terminal-only;
    # self-reference inside the log would just be noise.
    with tee_stdio(config.run_dir / f"run{suffix}.log"):
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

        # --- Model + optimizer ---
        print(f"building model on {device} (value_head={config.arch.value_head_variant})")
        state = make_state(device)
        n_params = sum(p.numel() for p in state.model.parameters())
        print(f"  params: {n_params:,}")

        # Resolve precision for the startup banner; `train_loop` re-derives
        # the autocast dtype from the same (deterministic) decision.
        resolved_precision = resolve_precision(config.precision, device)
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
        # Resume-only: announce the legacy cold-restart LR ramp so it's visible
        # in run{suffix}.log (the per-batch lr lands in batches.jsonl). Prints
        # only when a warmup is configured — silent on fresh/combined runs.
        if state.warmup is not None:
            w = state.warmup
            print(
                f"  legacy-resume LR warmup: ramping over first {w.n_batches} batches "
                f"({w.target_lr / w.n_batches:.2e} -> {w.target_lr:.2e})"
            )
        print()

        artifacts = RunArtifacts(
            run_dir=config.run_dir,
            ckpt_dir=ckpt_dir,
            flops_per_sample=flops_per_sample,
            peak_tflops=peak_tflops,
            suffix=suffix,
        )
        train_loop(state, config, loader, ds, val_samples, device, artifacts)


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


def train_loop(
    state: TrainingState,
    config: TrainConfig,
    loader: DataLoader,
    dataset: IterableDataset,
    val_samples: list[tuple[Path, int]],
    device: torch.device,
    artifacts: RunArtifacts,
) -> None:
    """Drive the epoch loop over a prepared `TrainingState` + `RunArtifacts`.

    Per epoch: train pass via `train_one_epoch`, full val pass via
    `run_val`, checkpoint save, one row to `epochs.jsonl`, console summary.
    The epoch range starts at `state.epoch + 1`, so a fresh state (epoch 0)
    runs `1..N` and a resumed state (epoch K) continues at `K+1` — epoch
    numbering stays monotonic across resume segments.

    JSONL handles (owned by `RunArtifacts`) are line-buffered and closed on
    context exit so `tail -f` works and a mid-epoch raise still flushes
    records to disk. The caller builds `state` + `artifacts` and measures
    FLOPs; this function only loops.
    """
    # Mixed-precision resolution. `amp_dtype is None` is the canonical
    # "AMP off" sentinel — autocast is wired through both code paths so the
    # fp32 case is a no-op rather than a separate branch. Mirrors the
    # GradScaler decision on `state` (both derive from the same precision).
    amp_dtype = torch.float16 if resolve_precision(config.precision, device) == "fp16" else None

    run_start = time.perf_counter()
    # `RunArtifacts` opens/closes the JSONL writers; the gpu sidecar is a
    # sibling context manager. Both unwind on a mid-epoch raise, flushing
    # the line-buffered records to disk.
    with artifacts, gpu_util_sidecar(artifacts.gpu_util_path, device):
        for epoch in range(state.epoch + 1, config.epochs + 1):
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
                warmup=state.warmup,
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
                    obs_cfg=config.arch.obs,
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
