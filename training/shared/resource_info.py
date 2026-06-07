"""Startup logging of the container's hardware envelope + the data pipeline's
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
The same hardware envelope is also dumped to `host_info.json` (structured, for
`bc.run_report`) — Modal places each run on a different cloud/host, so the
silicon behind a run varies and a slow draw is only diagnosable if recorded.

All Linux-specific reads (`/sys/fs/cgroup`, `/proc`, `/dev/shm`) are guarded,
so this is a safe partial no-op on a local macOS/MPS box.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform

import torch

from utils.format import format_bytes


def _read_int(path: str) -> int | None:
    """First whitespace-delimited int in a file, or None if unreadable."""
    try:
        return int(Path(path).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _read_str(path: str) -> str | None:
    """Stripped file contents, or None if unreadable/empty."""
    try:
        return Path(path).read_text().strip() or None
    except OSError:
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


def _cpuinfo_field(key: str) -> str | None:
    """First value for a `/proc/cpuinfo` key like `model name`."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith(key) and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _numa_node_count() -> int | None:
    """Number of NUMA nodes (`/sys/devices/system/node/nodeN`). NUMA crossing
    is a prime suspect for the DataLoader memory-bandwidth wall, so worth
    recording per host."""
    try:
        nodes = list(Path("/sys/devices/system/node").glob("node[0-9]*"))
        return len(nodes) or None
    except OSError:
        return None


def _virt_hint() -> str | None:
    """Best-effort instance/hypervisor hint from DMI (often unreadable in a
    container, hence guarded)."""
    for field in ("product_name", "sys_vendor"):
        val = _read_str(f"/sys/class/dmi/id/{field}")
        if val:
            return val
    return None


def _gpu_pcie(device: torch.device) -> tuple[int, int] | None:
    """(current PCIe link generation, width) for the CUDA device via NVML, or
    None off-CUDA / when NVML is unavailable. The link gen (gen4 vs gen5)
    directly sets the H2D copy bandwidth, which varies host to host."""
    if device.type != "cuda":
        return None
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
        gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
        width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
        return int(gen), int(width)
    except Exception:
        return None


def _modal_env() -> dict[str, str]:
    """All `MODAL_*` env vars — placement identity beyond the masked
    hostname (region, task id, etc.)."""
    return {k: v for k, v in os.environ.items() if k.startswith("MODAL_")}


def host_info(device: torch.device) -> dict:
    """The container's hardware envelope as a structured dict — the machine
    fingerprint behind a run. Every field is best-effort and may be None on a
    non-Linux/local box. Dumped to `host_info.json` and consumed by
    `bc.run_report`'s host-draw table."""
    getaffinity = getattr(os, "sched_getaffinity", None)
    affinity = len(getaffinity(0)) if getaffinity is not None else None
    shm = _shm_bytes()
    total_kb = _meminfo_kb("MemTotal")
    avail_kb = _meminfo_kb("MemAvailable")
    pcie = _gpu_pcie(device)

    gpu_name: str | None = None
    gpu_mem: int | None = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu_name, gpu_mem = props.name, props.total_memory

    return {
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "cpu_model": _cpuinfo_field("model name"),
        "numa_nodes": _numa_node_count(),
        "ram_total_bytes": total_kb * 1024 if total_kb else None,
        "ram_available_bytes": avail_kb * 1024 if avail_kb else None,
        "cgroup_mem_limit_bytes": _cgroup_mem_limit_bytes(),
        "shm_total_bytes": shm[0] if shm else None,
        "shm_avail_bytes": shm[1] if shm else None,
        "gpu_name": gpu_name,
        "gpu_mem_bytes": gpu_mem,
        "gpu_pcie_gen": pcie[0] if pcie else None,
        "gpu_pcie_width": pcie[1] if pcie else None,
        "virt": _virt_hint(),
        "kernel": platform.release(),
        "modal_env": _modal_env(),
    }


def log_resource_info(
    *,
    device: torch.device,
    batch_size: int,
    obs_channels: int,
    obs_dtype: str,
    spatial: tuple[int, int],
    num_workers: int,
    prefetch_factor: int | None,
    pin_memory: bool,
    run_dir: Path | None = None,
) -> None:
    """Print the container resource envelope + DataLoader in-flight footprint,
    and (when `run_dir` is given) dump the envelope to `host_info.json`.

    Called once at startup (from `build_dataloader`, under the run-log tee).
    `spatial` is the padded (H, W); `obs_dtype` ("fp32"/"fp16") sets the
    per-element size for the footprint math.
    """
    info = host_info(device)
    if run_dir is not None:
        (run_dir / "host_info.json").write_text(json.dumps(info, indent=2) + "\n")

    print("container resources:")

    cpu_extra = ""
    if info["cpu_model"]:
        cpu_extra += f"  model={info['cpu_model']}"
    if info["numa_nodes"]:
        cpu_extra += f"  numa_nodes={info['numa_nodes']}"
    print(
        f"  cpu: os.cpu_count()={info['cpu_count']}  "
        f"affinity={info['cpu_affinity']}{cpu_extra}"
    )

    if info["shm_total_bytes"] is not None:
        print(
            f"  /dev/shm: {format_bytes(info['shm_total_bytes'])} total "
            f"({format_bytes(info['shm_avail_bytes'])} free)"
        )
    if info["cgroup_mem_limit_bytes"] is not None:
        print(f"  cgroup memory limit: {format_bytes(info['cgroup_mem_limit_bytes'])}")
    if info["ram_total_bytes"] is not None:
        avail = info["ram_available_bytes"]
        avail_str = f"  available={format_bytes(avail)}" if avail else ""
        print(f"  system RAM: total={format_bytes(info['ram_total_bytes'])}{avail_str}")

    if device.type == "cuda":
        pcie = ""
        if info["gpu_pcie_gen"]:
            pcie = f"  pcie gen{info['gpu_pcie_gen']} x{info['gpu_pcie_width']}"
        print(f"  gpu: {info['gpu_name']} ({format_bytes(info['gpu_mem_bytes'])}){pcie}")
    else:
        print(f"  device: {device.type}")
    if info["virt"]:
        print(f"  virt: {info['virt']}  kernel: {info['kernel']}")

    # --- pipeline host-memory footprint ---
    h, w = spatial
    bytes_per_el = {"fp16": 2, "fp32": 4}[obs_dtype]
    per_batch = batch_size * obs_channels * h * w * bytes_per_el
    in_flight_batches = (
        num_workers * (prefetch_factor or 1) if num_workers > 0 else 1
    )
    print("pipeline host-memory footprint:")
    print(
        f"  per-batch obs: {format_bytes(per_batch)}  "
        f"(bs={batch_size} × {obs_channels}ch × {h}×{w} × {obs_dtype})"
    )
    detail = (
        f"{num_workers} workers × prefetch {prefetch_factor or 1} "
        f"= {in_flight_batches} batches"
        if num_workers > 0
        else "single-process (num_workers=0)"
    )
    pin_note = "; +pin_memory page-locked copies" if pin_memory else ""
    print(
        f"  in-flight prefetch: ~{format_bytes(per_batch * in_flight_batches)}  "
        f"({detail}{pin_note})"
    )
