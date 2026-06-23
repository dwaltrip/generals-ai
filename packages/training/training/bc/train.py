"""BC training runner.

Takes a validated `TrainConfig`, drives an end-to-end training run.
Pure runner: no CLI, no environment-specific defaults. CLI scaffolding
lives in `bc.train_cli`; the config dataclass + run-id factory live in
`bc.train_config`; entry-point wrappers in `packages/training/scripts/`.

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
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from training.bc.checkpoint import ckpt_name
from training.bc.constants import H_PADDED, W_PADDED
from training.bc.dataset import IterableDataset, assert_safe_loader, timed_collate
from training.bc.eval import FrameRecordCapture, dump_path, run_val, save_dump
from training.bc.loss import LossAccumulator, LossConfig, bc_loss
from training.bc.model import BCModel
from training.bc.resume_warmup import WarmupSchedule
from training.bc.run_dir import RunArtifacts
from training.bc.run_logger import RunLogger
from training.bc.splits import load_manifest, samples_for_split
from training.bc.state import TrainingState
from training.bc.train_config import TrainConfig
from training.shared.device import (
    dataloader_kwargs,
    disable_mps_fallback,
    move_batch,
    obs_for_model,
    pick_device,
    resolve_precision,
)
from training.shared.gpu_sidecar import gpu_util_sidecar
from training.shared.perf import compute_mfu, measure_total_flops, peak_tflops_fp32
from training.shared.resource_info import log_resource_info
from training.shared.timing import timer
from training.shared.timing_run import active_sink
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


def _move_batch_timed(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """`move_batch`, timed under `h2d`. On CUDA the copy is async (pinned +
    `non_blocking`), so wall-clock would catch only the launch — CUDA events
    time the real transfer; elsewhere a wall-clock `section` is the transfer.
    The event path adds a per-batch sync, acceptable since the loop already
    syncs each batch via the `.item()` logging. Inert when profiling is off."""
    if timer.enabled and device.type == "cuda":
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        moved = move_batch(batch, device)
        ev1.record()
        ev1.synchronize()
        timer.add("h2d", int(ev0.elapsed_time(ev1) * 1e6))  # ms → ns
        return moved
    with timer.section("h2d"):
        return move_batch(batch, device)


def train_one_epoch(
    epoch: int,
    model: BCModel,
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
    loss_cfg: LossConfig,
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

    acc = LossAccumulator(loss_cfg, model.active_aux_specs)
    epoch_start = time.perf_counter()
    n_batches_seen = 0

    # Rolling samples/sec — instantaneous rate across the last
    # log-every window. Reset every print so the number tracks
    # current throughput rather than smoothing over the epoch.
    window_start = epoch_start
    window_samples = 0

    # `fetch_wait` brackets the gap between batches — time the main loop is
    # blocked on the producer (the starvation signal). Manual start/stop because
    # the wait spans the `for`'s implicit `next()`. No-ops when profiling is off.
    timer.start("fetch_wait")
    for batch_idx, batch in enumerate(loader):
        timer.stop("fetch_wait")
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Legacy-resume LR warmup: set this batch's LR on the optimizer before
        # the step. No-op when warmup is None (fresh runs / combined resumes).
        if warmup is not None:
            warmup.step(optim)

        batch = _move_batch_timed(batch, device)
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
            out = model(obs_for_model(batch, amp_dtype), batch["valid_mask"])
            losses = bc_loss(out, batch, loss_cfg)
        scaler.scale(losses["total"]).backward()
        scaler.step(optim)
        scaler.update()

        B = batch["obs"].shape[0]
        acc.update(losses, batch_size=B)
        window_samples += B
        n_batches_seen += 1

        record = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "batch_size": B,
            "policy": float(losses["policy"].item()),
            "value": float(losses["value"].item()),
            "value_soft": float(losses["value_soft"].item()),
            "pass": float(losses["pass"].item()),
            "total": float(losses["total"].item()),
            "lr": optim.param_groups[0]["lr"],
            "n_non_pass": int(losses["n_non_pass"].item()),
            "wall_time_sec": round(time.perf_counter() - run_start, 3),
        }
        # Elim keys appear only when the head is built — keep non-elim rows
        # byte-identical. The two variants are mutually exclusive.
        if "elim" in losses:
            record["elim"] = float(losses["elim"].item())
            record["elim_soft"] = float(losses["elim_soft"].item())
            record["n_elim"] = int(losses["n_elim"].item())
        if "next_elim" in losses:
            record["next_elim"] = float(losses["next_elim"].item())
            record["n_next_elim"] = int(losses["n_next_elim"].item())
        logger.log_batch(record)

        if (batch_idx + 1) % log_every == 0:
            rate = window_samples / (time.perf_counter() - window_start)
            elim_str = (
                f"elim {losses['elim'].item():6.4f} " if "elim" in losses else ""
            )
            if "next_elim" in losses:
                elim_str = f"next_elim {losses['next_elim'].item():6.4f} "
            print(
                f"[epoch {epoch}] batch {batch_idx + 1} | "
                f"policy {losses['policy'].item():6.4f} "
                f"value {losses['value'].item():6.4f} "
                f"pass {losses['pass'].item():6.4f} "
                f"{elim_str}"
                f"total {losses['total'].item():6.4f} | "
                f"{rate:.0f} samples/sec"
            )
            window_start = time.perf_counter()
            window_samples = 0

        timer.start("fetch_wait")
    else:
        # Loop ran to exhaustion (no max-batches break): close the wait that
        # hit StopIteration so no timer is left open at snapshot.
        timer.stop("fetch_wait")

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
        prof_sink=active_sink(),
        elim_bin_edges=(
            config.arch.elim_bin_edges
            if config.arch.elim_head_variant is not None
            else None
        ),
        elim_head_variant=config.arch.elim_head_variant,
    )
    dl_kwargs = dataloader_kwargs(
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        device=device,
    )
    loader = DataLoader(
        ds, batch_size=config.batch_size, collate_fn=timed_collate, **dl_kwargs
    )
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
        obs_dtype=config.arch.obs.obs_dtype,
        spatial=(H_PADDED, W_PADDED),
        num_workers=dl_kwargs["num_workers"],
        prefetch_factor=dl_kwargs.get("prefetch_factor"),
        pin_memory=dl_kwargs["pin_memory"],
        run_dir=config.run_dir,
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
        # `begin()` enabled the timer before this call when --profile is set;
        # surface it here since the report prints after the tee closes.
        if timer.enabled:
            print(
                "profiling: ON — obs-pipeline timing seams active; "
                f"report → {config.run_dir / 'prof' / 'summary.json'}"
            )
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
    if "elim" in summary:
        elim_mean = f"  elim {summary['elim']:.4f}"
    elif "next_elim" in summary:
        elim_mean = f"  next_elim {summary['next_elim']:.4f}"
    else:
        elim_mean = ""
    print(
        f"[epoch {epoch}] mean: "
        f"policy {summary['policy']:.4f}  "
        f"value {summary['value']:.4f}  "
        f"pass {summary['pass']:.4f}"
        f"{elim_mean}  |  "
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
        ent = val_summary["policy_entropy"]
        ent_str = f"H {ent:.3f} (e^H {math.exp(ent):.1f})" if ent is not None else "H n/a"
        print(
            f"[epoch {epoch}] val | "
            f"top1 {_fmt_metric(val_summary['top1'])}  "
            f"top3 {_fmt_metric(val_summary['top3'])}  "
            f"pass_acc {_fmt_metric(val_summary['pass_acc'])}  "
            f"pass_frac {_fmt_metric(val_summary['pass_frac'])}  "
            f"{ent_str}"
        )
        # Elim head health: soft CE vs its soft-marginal floor (positive margin
        # = beats the constant-predictor baseline), top-1 bin acc, and the
        # prediction-entropy collapse alarm. Present only when the head is on.
        if val_summary.get("elim_soft") is not None:
            floor = val_summary["elim_soft_floor"]
            print(
                f"[epoch {epoch}] val | elim "
                f"soft {val_summary['elim_soft']:.4f} "
                f"(floor {floor:.4f}, margin {floor - val_summary['elim_soft']:+.4f})  "
                f"top1 {_fmt_metric(val_summary['elim_top1'])}  "
                f"H {val_summary['elim_pred_entropy']:.3f}"
            )
        # who-dies-next: only the loss is surfaced in-loop (accuracy / ramp /
        # horizon reads are offline from the dump).
        if val_summary.get("next_elim") is not None:
            print(
                f"[epoch {epoch}] val | next_elim CE {val_summary['next_elim']:.4f}"
            )
    else:
        print(f"[epoch {epoch}] val skipped (--skip-val)")


def write_val_dump(
    capture: FrameRecordCapture,
    config: TrainConfig,
    epoch: int,
    n_val_perspectives: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    val_summary: dict,
) -> None:
    """Persist one epoch's stratified val dump and print the cross-check line.

    The dump rides the same forward pass as `val_summary`, so the two mean
    value CEs should agree to float-accumulation noise — the printed diff is
    the in-training analog of the offline harness's check against
    `epochs.jsonl`. A large diff means the capture and the loss path have
    diverged.
    """
    records = capture.finalize()
    path = dump_path(config.run_dir, epoch)
    meta = {
        "run_dir": str(config.run_dir),
        "checkpoint": ckpt_name(epoch),
        "epoch": epoch,
        # In-training dumps always walk the full val split; `sample_frac` /
        # `n_perspectives*` keep the offline dump's meta shape so the report
        # consumer reads either.
        "sample_frac": 1.0,
        "seed": config.seed,
        "n_perspectives": n_val_perspectives,
        "n_perspectives_total": n_val_perspectives,
        "device": device.type,
        "producer": "train",
        # Under AMP the records derive from an fp16 forward (offline dumps
        # are fp32-forward) — the comparability caveat lives here.
        "forward_dtype": "fp16" if amp_dtype is not None else "fp32",
    }
    # Bin edges are a time_bin-only concept (the offline report needs them to
    # interpret the bins); a residual variant-specific dump-meta site, not part of
    # the per-frame spec dispatch.
    if config.arch.elim_head_variant == "time_bin":
        meta["elim_bin_edges"] = list(config.arch.elim_bin_edges)
    save_dump(records, path, meta)
    dump_value = float(records["value_ce"].mean())
    diff = dump_value - val_summary["value"]
    msg = (
        f"[epoch {epoch}] val dump: {capture.n_frames:,} frames -> "
        f"analysis/{path.name} | mean value CE {dump_value:.6g} "
        f"vs val {val_summary['value']:.6g} (diff {diff:+.2e})"
    )
    if "elim_ce" in records:
        # Same correctness check for the elim column: dump's masked-mean hard CE
        # vs the recorded val `elim`. Diverges only if capture and loss disagree.
        alive = records["alive_mask"]
        dump_elim = float(records["elim_ce"][alive].mean())
        msg += f" | mean elim CE {dump_elim:.6g} vs val {val_summary['elim']:.6g}"
    if "next_elim_ce" in records:
        # Same check for the who-dies-next column: dump's mean CE over frames with
        # a defined next victim vs the recorded val `next_elim`.
        kept = records["next_elim_target"] != -1
        dump_next = float(records["next_elim_ce"][kept].mean())
        msg += (
            f" | mean next-elim CE {dump_next:.6g} "
            f"vs val {val_summary['next_elim']:.6g}"
        )
    print(msg)


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

    loss_cfg = config.loss

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
                loss_cfg=loss_cfg,
                warmup=state.warmup,
            )
            if config.skip_val:
                val_summary = None
            else:
                capture = FrameRecordCapture() if config.dump_val_frames else None
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
                    loss_cfg=loss_cfg,
                    capture=capture,
                    elim_bin_edges=(
                        config.arch.elim_bin_edges
                        if config.arch.elim_head_variant is not None else None
                    ),
                    elim_head_variant=config.arch.elim_head_variant,
                )
                if capture is not None:
                    write_val_dump(
                        capture, config, epoch, len(val_samples),
                        device, amp_dtype, val_summary,
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
