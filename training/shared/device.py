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
    """Move every tensor in a flat dict-of-tensors batch onto `device`."""
    return {k: v.to(device) for k, v in batch.items()}


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
