"""Startup logging of the container's resource envelope + the data pipeline's
host-memory footprint.

Training feeds the GPU from host RAM: DataLoader workers build obs tensors,
prefetch them, pin them, and copy them to the device. On a container those
host-side limits (CPU count, `/dev/shm`, the cgroup memory ceiling) are easy
to under-provision silently, and when the in-flight buffer outgrows them the
GPU starves — visible only as a throughput drop, with nothing in the log to
explain it.

This prints, once at startup, what the container actually got next to an
estimate of the DataLoader's in-flight footprint, so a starvation regression
shows up on line one of `run.log` instead of being reverse-engineered later.

All Linux-specific reads (`/sys/fs/cgroup`, `/proc`, `/dev/shm`) are guarded,
so this is a safe partial no-op on a local macOS/MPS box.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def _fmt_bytes(n: float) -> str:
    """Human-readable IEC size (e.g. `252.0 MiB`)."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    raise AssertionError("unreachable")  # the TiB branch always returns


def _read_int(path: str) -> int | None:
    """First whitespace-delimited int in a file, or None if unreadable."""
    try:
        return int(Path(path).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _cgroup_mem_limit_bytes() -> int | None:
    """Container memory ceiling from cgroup v2 then v1. None = no limit set
    (v2 `max` sentinel) or the files aren't present (non-Linux)."""
    try:
        raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        return None if raw == "max" else int(raw)
    except (OSError, ValueError):
        return _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")


def _meminfo_kb(key: str) -> int | None:
    """Value (in kB) for a `/proc/meminfo` key like `MemTotal`."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _shm_bytes() -> tuple[int, int] | None:
    """(total, available) bytes for `/dev/shm`, or None if not present."""
    try:
        st = os.statvfs("/dev/shm")
    except OSError:
        return None
    return st.f_blocks * st.f_frsize, st.f_bavail * st.f_frsize


def log_resource_info(
    *,
    device: torch.device,
    batch_size: int,
    obs_channels: int,
    spatial: tuple[int, int],
    num_workers: int,
    prefetch_factor: int | None,
    pin_memory: bool,
) -> None:
    """Print the container resource envelope + DataLoader in-flight footprint.

    Called once at startup (from `build_dataloader`, under the run-log tee).
    `spatial` is the padded (H, W); obs tensors are fp32 (4 bytes/element).
    """
    print("container resources:")

    # sched_getaffinity (the usable-core count) is Linux-only; absent on macOS.
    getaffinity = getattr(os, "sched_getaffinity", None)
    affinity = len(getaffinity(0)) if getaffinity is not None else None
    print(f"  cpu: os.cpu_count()={os.cpu_count()}  affinity={affinity}")

    shm = _shm_bytes()
    if shm is not None:
        total, avail = shm
        print(f"  /dev/shm: {_fmt_bytes(total)} total ({_fmt_bytes(avail)} free)")

    lim = _cgroup_mem_limit_bytes()
    if lim is not None:
        print(f"  cgroup memory limit: {_fmt_bytes(lim)}")

    total_kb = _meminfo_kb("MemTotal")
    if total_kb is not None:
        avail_kb = _meminfo_kb("MemAvailable")
        avail_str = f"  available={_fmt_bytes(avail_kb * 1024)}" if avail_kb else ""
        print(f"  system RAM: total={_fmt_bytes(total_kb * 1024)}{avail_str}")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"  gpu: {props.name} ({_fmt_bytes(props.total_memory)})")
    else:
        print(f"  device: {device.type}")

    # --- pipeline host-memory footprint ---
    h, w = spatial
    per_batch = batch_size * obs_channels * h * w * 4  # fp32 obs tensor
    in_flight_batches = (
        num_workers * (prefetch_factor or 1) if num_workers > 0 else 1
    )
    print("pipeline host-memory footprint:")
    print(
        f"  per-batch obs: {_fmt_bytes(per_batch)}  "
        f"(bs={batch_size} × {obs_channels}ch × {h}×{w} × fp32)"
    )
    detail = (
        f"{num_workers} workers × prefetch {prefetch_factor or 1} "
        f"= {in_flight_batches} batches"
        if num_workers > 0
        else "single-process (num_workers=0)"
    )
    pin_note = "; +pin_memory page-locked copies" if pin_memory else ""
    print(
        f"  in-flight prefetch: ~{_fmt_bytes(per_batch * in_flight_batches)}  "
        f"({detail}{pin_note})"
    )
