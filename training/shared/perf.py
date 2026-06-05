"""
Performance-measurement helpers: FLOPs counting + Model FLOPs Utilization (MFU).

## What MFU is

MFU answers "how close to the GPU's theoretical peak compute are we actually
running?" It's a ratio:

    MFU = (achieved FLOPS/sec) / (device peak FLOPS/sec)

For a training step, "achieved FLOPS/sec" is `samples_per_sec * FLOPs_per_sample`
(both forward + backward). The denominator depends on the device AND the
precision regime in use — T4's FP32 peak is 8.1 TFLOPS, but its FP16 tensor-core
peak is 65 TFLOPS, an 8× difference. A run hitting "32% of FP32 peak" might be
"4% of tensor-core peak" — same throughput, different framing.

## Why care

Without MFU, sps numbers are uninterpretable in isolation. "237 sps" reads
totally differently if it's 5% of peak (huge headroom) vs. 50% (near the
ceiling). MFU turns every run into a self-grading data point.

Common rules of thumb:
  - Frontier-shop transformer training:   40–55% MFU.
  - Hobby / small-CNN training:           15–35% MFU is normal.
  - <10% MFU on a substantive workload:   probably data-starved or
                                           kernel-launch-bound, not
                                           compute-bound.

## Peak lookup table is a moving target

`peak_tflops_fp32` hardcodes a small table of GPUs we've actually used or
are likely to. Unknown devices return `None` and MFU isn't computed (better
to omit than to compare against a wrong denominator). Add entries as needed.
"""

from __future__ import annotations

from collections.abc import Callable
import platform
import subprocess

import torch
from torch.utils.flop_counter import FlopCounterMode


def measure_total_flops(forward_call: Callable[[], torch.Tensor]) -> int:
    """Run a forward pass + backward pass under a FlopCounterMode, return total FLOPs.

    `forward_call` must be a zero-arg callable that performs the model forward
    pass and returns a *scalar* tensor suitable for `.backward()` (typically
    a summed surrogate loss). The helper handles the FlopCounterMode setup +
    backward() invocation.

    Returns: total counted FLOPs across forward + backward. Divide by the
    sample batch size to get FLOPs/sample.

    Why a callable rather than (model, input): the model's forward signature
    isn't standardized (some take a tensor, some take a dict), and the
    "loss" to backprop through can be a summed surrogate or a real loss.
    Passing the closure lets the caller control both.
    """
    with FlopCounterMode(display=False) as fc:
        loss = forward_call()
        loss.backward()
    return fc.get_total_flops()


# Peak FP32 TFLOPS for devices we've run on, or might. Keys are substring
# matches against `torch.cuda.get_device_name()` (for CUDA) or the macOS
# `sysctl machdep.cpu.brand_string` (for Apple Silicon). Unknown → None.
#
# Numbers are vendor-published "peak" — always optimistic. Real ceilings
# are typically 80-90% of these on a perfectly tuned workload. Good enough
# as a denominator for MFU sanity-checking; don't read MFU to two decimals.
_CUDA_PEAKS_FP32: dict[str, float] = {
    "Tesla T4":  8.1,
    "Tesla V100": 15.7,
    "A100":      19.5,
    "H100":      67.0,
    "L4":        30.3,
    "L40":       90.5,
    "RTX 4090":  83.0,
}

_APPLE_PEAKS_FP32: dict[str, float] = {
    # M1 Max has 24-core (7.8 TFLOPS) and 32-core (10.4 TFLOPS) variants.
    # The brand string doesn't distinguish them; we go with the 32-core
    # number on the assumption that's the variant in active use here. If
    # MFU on M1 looks suspiciously high, this might be why.
    "Apple M1 Max": 10.4,
    "Apple M2 Max": 13.6,
    "Apple M3 Max": 14.2,
    "Apple M4 Max": 18.4,
}

# Peak FP16 *tensor-core* TFLOPS, DENSE (no structured sparsity), FP32-accumulate
# — the regime PyTorch AMP fp16 training actually runs in, so this is the right
# MFU denominator for our fp16 runs. The fp32 table above is the WRONG one for
# fp16 work: tensor cores deliver ~16x the fp32-ALU peak, so MFU-vs-fp32 reads
# well over 100% once the GPU is actually busy.
#
# Dense, NOT the headline "with sparsity" figure (2x larger, irrelevant to our
# dense training). Sourced from NVIDIA datasheets / vendor product guides:
#   T4    65    (Turing — no sparsity; 65 is the FP16/FP32-mixed figure)
#   L4   121    (242 with sparsity — Lenovo/NVIDIA L4 product guide)
#   A100 312    (624 with sparsity — A100 datasheet)
#   H100 989.5  (1979 with sparsity — H100 SXM/HBM3 datasheet)
# Caveat: substring "H100" assumes the SXM/HBM3 part (what Modal provisions); the
# PCIe H100 is ~756 dense. Refine the key if a PCIe H100 ever shows up.
_CUDA_PEAKS_FP16: dict[str, float] = {
    "Tesla T4": 65.0,
    "L4":       121.0,
    "A100":     312.0,
    "H100":     989.5,
}


def _apple_chip_name() -> str | None:
    """Return e.g. `"Apple M1 Max"` on macOS, or None on other platforms / failure.

    Uses `sysctl machdep.cpu.brand_string`. Python's stdlib doesn't expose
    this cleanly across darwin versions; subprocess is the reliable path.
    """
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2.0, check=True,
        )
        return result.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def peak_tflops_fp32(device: torch.device) -> float | None:
    """Look up peak FP32 TFLOPS for a device. Returns None on unknown.

    Logic:
      - CUDA: match `torch.cuda.get_device_name(index)` against substrings.
      - MPS:  read Apple chip name via sysctl, match against substrings.
      - CPU:  None (MFU on CPU isn't a useful framing for our work).
    """
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        name = torch.cuda.get_device_name(idx)
        for needle, tflops in _CUDA_PEAKS_FP32.items():
            if needle in name:
                return tflops
        return None
    if device.type == "mps":
        chip = _apple_chip_name()
        if chip is None:
            return None
        for needle, tflops in _APPLE_PEAKS_FP32.items():
            if needle in chip:
                return tflops
        return None
    return None


def peak_tflops_fp16(device: torch.device) -> float | None:
    """Peak dense FP16 tensor-core TFLOPS for a CUDA device, or None if unknown.

    CUDA-only by design: this is the MFU denominator for fp16 AMP training (what
    `precision=auto` selects on CUDA). MPS/CPU return None — there's no Apple
    tensor-core figure we'd trust as a denominator, and CPU MFU isn't a useful
    framing. Substring-matched against `get_device_name`, same as the fp32 table.
    """
    if device.type != "cuda":
        return None
    idx = device.index if device.index is not None else 0
    name = torch.cuda.get_device_name(idx)
    for needle, tflops in _CUDA_PEAKS_FP16.items():
        if needle in name:
            return tflops
    return None


def compute_mfu(
    sps: float,
    flops_per_sample: float,
    peak_tflops: float | None,
) -> float | None:
    """MFU as a fraction in [0, 1], or None if peak isn't known.

    `sps` is the training-step samples/sec (fwd+bwd). `flops_per_sample`
    is the model's total fwd+bwd FLOPs at batch size 1 (or batch_size B
    divided by B — same number). `peak_tflops` is what the device can
    achieve in the limit; pass `None` if unknown and we return `None`.
    """
    if peak_tflops is None or sps <= 0:
        return None
    achieved_tflops = (sps * flops_per_sample) / 1e12
    return achieved_tflops / peak_tflops
