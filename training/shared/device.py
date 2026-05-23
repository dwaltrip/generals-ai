"""
Torch device handling shared across training scripts: device selection,
batch transfer, MPS-fallback policy.
"""

from __future__ import annotations

import os

import torch


def pick_device(arg: str) -> torch.device:
    """
    Resolve `"auto" | "cuda" | "mps" | "cpu"` to a concrete `torch.device`.

    `"auto"` picks CUDA if available, then MPS, then CPU. Explicit choices
    raise if the requested accelerator isn't available — silent fallback to
    CPU on a cloud GPU box or an M-series Mac is a worse failure mode than
    a loud error at startup.
    """
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is not available")
        return torch.device("mps")
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"unknown device arg: {arg!r}")


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move every tensor in a flat dict-of-tensors batch onto `device`.

    `non_blocking=True` enables async H2D transfers when the source tensor
    lives in pinned memory and the target is CUDA — the copy is queued on
    the default CUDA stream and overlaps with GPU compute on the previous
    batch. On non-CUDA devices, or with unpinned source tensors, the flag
    is silently ignored, so it's safe to set unconditionally. See
    `TrainConfig.pin_memory` for the full picture on why this matters.

    Caveat for future callers: with `non_blocking=True`, the returned
    tensors are *not* guaranteed to be host-readable immediately. Any
    code that does `.cpu()`, `.numpy()`, or `.item()` on a moved tensor
    before a CUDA op consumes it on the same stream will race against
    the in-flight copy and may read stale memory. The current call sites
    (train + eval) feed the batch directly into `model(...)` on the
    default stream, which CUDA orders correctly. Add `torch.cuda.synchronize()`
    if you need a host-side read.
    """
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def dataloader_kwargs(
    *,
    num_workers: int,
    pin_memory: bool | None,
    prefetch_factor: int,
    device: torch.device,
) -> dict:
    """Resolve the device-dependent DataLoader kwargs for our training
    loop. Returns a dict the caller `**`-unpacks into `DataLoader(...)`.

    - `pin_memory=None` (auto) → `True` iff the target device is CUDA;
      explicit True/False overrides. See `TrainConfig.pin_memory` for the
      rationale behind the auto-default.
    - `prefetch_factor` is omitted from the returned dict when
      `num_workers == 0` because PyTorch rejects it in that mode.

    Centralized so the train and val loaders apply the same resolution —
    a tweak to either rule lands in one place instead of two. The caller
    keeps control of the actual `DataLoader` construction and can read
    the resolved `pin_memory` straight out of the returned dict for
    logging/provenance.
    """
    kwargs: dict = {
        "num_workers": num_workers,
        "pin_memory": pin_memory if pin_memory is not None else device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def resolve_precision(arg: str, device: torch.device) -> str:
    """Resolve `"auto" | "fp32" | "fp16"` to a concrete precision string.

    `"auto"` chooses `"fp16"` if CUDA is the device, else `"fp32"`. The
    asymmetry is deliberate: CUDA tensor cores make FP16 a near-mandatory
    win on real GPU training; MPS's autocast is less battle-tested and
    we don't have evidence of a big win on our model, so we default
    conservative and let the user opt in explicitly.

    Explicit `"fp16"` on a non-CUDA device is allowed (autocast still
    runs) but won't unlock the tensor-core ceiling that motivates AMP
    in the first place.
    """
    valid = ("auto", "fp32", "fp16")
    if arg not in valid:
        raise ValueError(f"unknown precision: {arg!r}, expected one of {valid}")
    if arg == "auto":
        return "fp16" if device.type == "cuda" else "fp32"
    return arg


def disable_mps_fallback() -> None:
    """
    Unset `PYTORCH_ENABLE_MPS_FALLBACK` so unsupported MPS ops raise loudly.

    Silent CPU fallback is a footgun on M-series boxes: a long training run
    can secretly degrade to CPU for one bad op and tank throughput without
    any error. Call this at the top of any script that does real model work
    on MPS. Idempotent — safe to call when the env var is already unset.
    """
    had_fallback = os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    if had_fallback is not None:
        print(
            f"NOTE: PYTORCH_ENABLE_MPS_FALLBACK was set in env ({had_fallback!r}); "
            "unset for this run so unsupported MPS ops error loudly."
        )
