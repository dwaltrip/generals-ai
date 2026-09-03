from __future__ import annotations

import hashlib

import numpy as np

from training.bc.datapipe.walk import WalkFrame


# Hash: blake2b, 8-byte digest, read as little-endian uint64. Canonical bytes
# are the array's C-order buffer in its emitted dtype. TODO: sha256 may be
# faster on this machine (7.15-2 §2); pick before the first committed bless.


def _hash64(buf: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(buf, digest_size=8).digest(), "little")


def _hash_along_first_axis(stack: np.ndarray) -> np.ndarray:
    stack = np.ascontiguousarray(stack)
    return np.array([_hash64(stack[i].tobytes()) for i in range(stack.shape[0])], dtype=np.uint64)


def hash_frames(frames: list[WalkFrame]) -> np.ndarray:
    return _hash_along_first_axis(np.stack([f.obs for f in frames]))


def hash_channels(frames: list[WalkFrame]) -> np.ndarray:
    stack = np.stack([f.obs for f in frames])  # [T, C, H, W]
    return _hash_along_first_axis(np.moveaxis(stack, 1, 0))


def hash_mask_frames(masks: np.ndarray) -> np.ndarray:
    return _hash_along_first_axis(masks)
