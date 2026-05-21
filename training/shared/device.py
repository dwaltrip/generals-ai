"""
Torch device handling shared across training scripts: device selection,
batch transfer, MPS-fallback policy.
"""

from __future__ import annotations

import os

import torch


def pick_device(arg: str) -> str:
    """
    Resolve `"auto" | "mps" | "cpu"` to a concrete device string.

    `"auto"` picks MPS if available, else CPU. CUDA isn't considered — add
    a branch here when we start running on CUDA boxes.
    """
    if arg != "auto":
        return arg
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def move_batch(
    batch: dict[str, torch.Tensor], device: str
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
