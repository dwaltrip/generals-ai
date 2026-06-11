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


def is_non_blocking_safe_device(device: torch.device) -> bool:
    """Whether async H2D copies (`non_blocking=True`) are safe on `device`.

    True only on CUDA, where the semantics are well-defined: with a pinned
    source the copy is queued on the default stream, and stream ordering
    guarantees the GPU sees the data before any op consumes it.

    On MPS the flag is NOT ignored (despite older torch docs implying
    non-CUDA devices drop it): the copy really is async, and the CPU source
    tensor must outlive it. `move_batch` callers rebind their only batch
    reference (`batch = move_batch(batch, ...)`), freeing the host buffers
    while the copy may still be in flight — a use-after-free race observed
    on torch 2.12 / M1 Max as corrupted batch tensors (garbage values,
    impossible sums) and libmalloc invalid-free aborts. It surfaced in
    `run_val` but the train loop shares the same hazard; blocking copies
    on MPS close it for both.
    """
    return device.type == "cuda"


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move every tensor in a flat dict-of-tensors batch onto `device`.

    H2D copies are async only where that's safe — see
    `is_non_blocking_safe_device`. On CUDA with pinned source memory the
    async copy overlaps with GPU compute on the previous batch; see
    `TrainConfig.pin_memory` for the full picture on why this matters.

    Caveat for CUDA callers: with the async path, the returned tensors are
    *not* guaranteed to be host-readable immediately. Any code that does
    `.cpu()`, `.numpy()`, or `.item()` on a moved tensor before a CUDA op
    consumes it on the same stream will race against the in-flight copy
    and may read stale memory. The current call sites (train + eval) feed
    the batch directly into `model(...)` on the default stream, which CUDA
    orders correctly. Add `torch.cuda.synchronize()` if you need a
    host-side read.
    """
    non_blocking = is_non_blocking_safe_device(device)
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


def obs_for_model(
    batch: dict[str, torch.Tensor], amp_dtype: torch.dtype | None
) -> torch.Tensor:
    """The obs tensor in the dtype the model's first conv expects.

    Obs is built fp16 or fp32 (`ObsConfig.obs_dtype`) to control bytes on the
    handoff path; the model's compute dtype is a separate axis. Under autocast
    (`amp_dtype` set), the conv casts its own input, so fp16 obs flows straight
    in (and fp32 obs would be cast down anyway) — pass it through. With autocast
    off (`amp_dtype is None`: fp32 / MPS), upcast here, on-device after the
    cheaper fp16 h2d, so fp16 obs doesn't dtype-clash with fp32 weights. `.float()`
    is a no-op when obs is already fp32, so this is inert on the all-fp32 path.
    """
    obs = batch["obs"]
    return obs if amp_dtype is not None else obs.float()


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
