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
  - `checkpoints/epoch_NNN.pt` — model state_dict at end of each epoch.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

import torch
from torch.utils.data import DataLoader

from bc.constants import H_PADDED, OBS_CHANNELS, W_PADDED
from bc.dataset import IterableDataset, assert_safe_loader
from bc.eval import run_val
from bc.loss import LossAccumulator, bc_loss
from bc.model import BCModel
from bc.splits import load_manifest, samples_for_split
from bc.train_config import TrainConfig, json_default
from shared.device import (
    dataloader_kwargs,
    disable_mps_fallback,
    move_batch,
    pick_device,
    resolve_precision,
)
from shared.gpu_sidecar import gpu_util_sidecar
from shared.log import tee_stdio
from shared.perf import compute_mfu, measure_total_flops, peak_tflops_fp32


def _write_jsonl(fp: TextIO, record: dict) -> None:
    """Append one record + newline. Files are opened line-buffered, so a
    `tail -f` sees each record as soon as the newline lands."""
    fp.write(json.dumps(record) + "\n")


def _fmt_metric(x: float | None, prec: int = 4) -> str:
    """Format a metric for console output, with `n/a` fallback for `None`.
    `run_val` returns `None` for accuracies whose denominators are 0."""
    return f"{x:.{prec}f}" if x is not None else "n/a"


def _ckpt_name(epoch: int) -> str:
    """Deterministic checkpoint filename for an epoch. Split from
    `_save_checkpoint` so callers can refer to the planned name in logs
    or `epochs.jsonl` records *before* the save actually executes —
    used to preserve epoch metrics if the save itself raises.
    """
    return f"epoch_{epoch:03d}.pt"


def _save_checkpoint(model: torch.nn.Module, ckpt_dir: Path, epoch: int) -> str:
    """Save the model's `state_dict` to `<ckpt_dir>/epoch_NNN.pt`. Returns
    the filename (sans dir) for logging.

    State-dict only — no optim/RNG/epoch payload. Resume-from-checkpoint
    isn't on the spike's path; reloading for eval or inference needs only
    the weights. Epoch number lives in the filename + `epochs.jsonl`.
    """
    name = _ckpt_name(epoch)
    torch.save(model.state_dict(), ckpt_dir / name)
    return name


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
        out = model(x)
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
    batches_fp: TextIO,
    run_start: float,
    max_batches: int | None,
    log_every: int,
    scaler: torch.amp.GradScaler,
    amp_dtype: torch.dtype | None,
) -> dict:
    """Run one epoch of BC training.

    Iterates `loader`, performs forward/backward/optim.step per batch,
    writes per-batch JSONL records to `batches_fp`, and prints a console
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
            out = model(batch["obs"])
            losses = bc_loss(out, batch)
        scaler.scale(losses["total"]).backward()
        scaler.step(optim)
        scaler.update()

        B = batch["obs"].shape[0]
        acc.update(losses, batch_size=B)
        window_samples += B
        n_batches_seen += 1

        _write_jsonl(batches_fp, {
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


def initialize_run_dir(config: TrainConfig) -> None:
    """Create `config.run_dir` and persist run provenance.

    Mkdirs the run dir with `exist_ok=False` so two runs landing in the
    same wall-clock second collide explicitly. Writes `args.json` (full
    `TrainConfig` as JSON) and announces the path. Call before `bc_run`.

    Split out from `bc_run` so cloud callers can drop sibling provenance
    files (e.g. `args_cloud.json`) into the run dir *before* training
    starts, instead of relying on a try/finally cleanup hook.
    """
    config.run_dir.mkdir(parents=True, exist_ok=False)
    print(f"run dir: {config.run_dir}")
    with (config.run_dir / "args.json").open("w") as fp:
        json.dump(asdict(config), fp, default=json_default, indent=2)


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
    # dataset summary, model build, training — lands in console.log. The
    # "run dir:" line printed by `initialize_run_dir` stays terminal-only;
    # self-reference inside the log would just be noise.
    with tee_stdio(config.run_dir / "console.log"):
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

        run_loop(
            config=config,
            device=device,
            dataset=ds,
            loader=loader,
            val_samples=val_samples,
            run_dir=config.run_dir,
            ckpt_dir=ckpt_dir,
        )


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
    summary. JSONL handles are line-buffered and closed in `finally` so
    `tail -f` works and a mid-epoch raise still flushes records to disk.
    """
    # --- Model + optimizer ---
    print(f"building model on {device}")
    model = BCModel().to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    # Mixed precision resolution. `amp_dtype is None` is the canonical
    # "AMP off" sentinel — autocast/scaler are wired through both code
    # paths so the fp32 case is a no-op rather than a separate branch.
    resolved_precision = resolve_precision(config.precision, device)
    amp_dtype = torch.float16 if resolved_precision == "fp16" else None
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is not None)
    print(f"  precision: {resolved_precision} (config={config.precision!r})")

    # FLOPs/sample + device peak: drives MFU per-epoch. FLOPs are measured
    # once on a synthetic batch; peak comes from a hardcoded device table
    # (`shared.perf`) and is `None` on unknown hardware — in which case
    # MFU is omitted rather than computed against a wrong denominator.
    flops_per_sample = _measure_model_flops_per_sample(model, device)
    peak_tflops = peak_tflops_fp32(device)
    print(f"  fwd+bwd FLOPs/sample: {flops_per_sample / 1e9:.2f} GFLOPs")
    if peak_tflops is not None:
        print(f"  device peak FP32:     {peak_tflops:.1f} TFLOPS")
    else:
        print("  device peak FP32:     unknown (MFU will be omitted)")
    print()

    # --- Train loop ---
    run_start = time.perf_counter()
    # `ExitStack` cleanly enables multiple context-manager-like helpers w/o deep nesting.
    with contextlib.ExitStack() as xs:
        # JSONL writers are line-buffered: `tail -f` sees each record as it's written
        batches_fp = xs.enter_context((run_dir / "batches.jsonl").open("w", buffering=1))
        epochs_fp = xs.enter_context((run_dir / "epochs.jsonl").open("w", buffering=1))
        xs.enter_context(gpu_util_sidecar(run_dir, device))

        for epoch in range(1, config.epochs + 1):
            summary = train_one_epoch(
                epoch=epoch,
                model=model,
                optim=optim,
                dataset=dataset,
                loader=loader,
                device=device,
                batches_fp=batches_fp,
                run_start=run_start,
                max_batches=config.max_batches,
                log_every=config.log_every,
                scaler=scaler,
                amp_dtype=amp_dtype,
            )
            val_summary = run_val(
                model=model,
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
            mfu = compute_mfu(summary["samples_per_sec"], flops_per_sample, peak_tflops)
            summary["mfu"] = round(mfu, 4) if mfu is not None else None
            summary["flops_per_sample"] = flops_per_sample

            # Write the epoch row + print the summary BEFORE saving the checkpoint.
            # A checkpoint-save failure shouldn't lose the epoch's computed metrics.
            ckpt_name = _ckpt_name(epoch)
            _write_jsonl(epochs_fp, {
                "epoch": epoch,
                **summary,
                "val": val_summary,
                "ckpt": ckpt_name,
            })
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

            # Now the actual save. Failure here raises out of the loop
            # with metrics + summary already persisted above.
            _save_checkpoint(model, ckpt_dir, epoch)
            print(f"[epoch {epoch}] saved checkpoint: checkpoints/{ckpt_name}")
            print()
