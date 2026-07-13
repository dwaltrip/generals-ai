"""Reference-layer builders over a walked obs stack: per-frame channel stats,
frame/channel hashes, and the provisional event-adjacent frozen-frame pick."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


STAT_SETS: dict[str, tuple[str, ...]] = {
    "base": ("nnz", "mean", "min", "max"),
    "ext": ("nnz", "mean", "min", "max", "p90", "p99"),
}


def channel_stats(obs: np.ndarray, stats: tuple[str, ...]) -> np.ndarray:
    """[C, len(stats)] float32 for one [C, H, W] frame."""
    flat = obs.reshape(obs.shape[0], -1).astype(np.float32)
    cols: list[np.ndarray] = []
    for s in stats:
        if s == "nnz":
            cols.append(np.count_nonzero(flat, axis=1).astype(np.float32))
        elif s == "mean":
            cols.append(flat.mean(axis=1))
        elif s == "min":
            cols.append(flat.min(axis=1))
        elif s == "max":
            cols.append(flat.max(axis=1))
        elif s.startswith("p"):
            cols.append(np.percentile(flat, float(s[1:]), axis=1).astype(np.float32))
        else:
            raise ValueError(f"unknown stat {s!r}")
    return np.stack(cols, axis=1)


def stack_stats(obs_stack: np.ndarray, stats: tuple[str, ...]) -> np.ndarray:
    """[T, C, len(stats)] float32 over a [T, C, H, W] stack."""
    return np.stack([channel_stats(frame, stats) for frame in obs_stack], axis=0)


def _hash64(buf: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(buf, digest_size=8).digest(), "little")


def frame_hashes(obs_stack: np.ndarray) -> np.ndarray:
    """[T] uint64 — one hash per frame over canonical bytes (C-order, stack dtype)."""
    stack = np.ascontiguousarray(obs_stack)
    return np.array(
        [_hash64(stack[t].tobytes()) for t in range(stack.shape[0])], dtype=np.uint64
    )


def frame_channel_hashes(obs_stack: np.ndarray) -> np.ndarray:
    """[T, C] uint64 — one hash per channel plane."""
    stack = np.ascontiguousarray(obs_stack)
    T, C = stack.shape[0], stack.shape[1]
    out = np.empty((T, C), dtype=np.uint64)
    for t in range(T):
        for c in range(C):
            out[t, c] = _hash64(stack[t, c].tobytes())
    return out


def pick_frozen_frames(sim: dict[str, np.ndarray], end_t: int) -> list[int]:
    """Provisional event-adjacent pick (probe-grade, for sizing only): t=0,
    capture ticks ±1, death ticks, the first production tick, the last walked
    frame."""
    picks = {0, end_t - 1}
    for tc in sim["capture_events"][:, 0]:
        picks.update((int(tc) - 1, int(tc), int(tc) + 1))
    picks.update(int(td) for td in sim["death_events"][:, 0])
    if end_t > 50:
        picks.add(50)
    return sorted(p for p in picks if 0 <= p < end_t)


def npz_compressed_bytes(path: Path, **arrays: np.ndarray) -> int:
    np.savez_compressed(path, **arrays)
    return path.stat().st_size


def npy_raw_bytes(path: Path, arr: np.ndarray) -> int:
    np.save(path, arr)
    return path.stat().st_size
