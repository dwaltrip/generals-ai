from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


# The hash is sha256 truncated to 8 bytes, read as a little-endian uint64, over
# the array's C-order bytes in its emitted dtype. Chosen over blake2b for speed
# (hardware SHA on the M1).


def _new_hasher():
    return hashlib.sha256()


def _as_uint64(digest: bytes) -> int:
    return int.from_bytes(digest[:8], "little")


def _hash64(buf: bytes) -> int:
    h = _new_hasher()
    h.update(buf)
    return _as_uint64(h.digest())


def hash_along_first_axis(stack: np.ndarray) -> np.ndarray:
    stack = np.ascontiguousarray(stack)
    return np.array([_hash64(stack[i].tobytes()) for i in range(stack.shape[0])], dtype=np.uint64)


@dataclass(frozen=True)
class ObsDigest:
    frame_hashes: np.ndarray    # [T] uint64
    channel_hashes: np.ndarray  # [C] uint64


class ObsHasher:
    # Iteratively update the hash one frame at a time. Otherwise, the frames
    # would be stacked in memory (up to a few hundred MB per walk).
    def __init__(self) -> None:
        self._frame_hashes: list[int] = []
        self._channel_hashers: list | None = None

    def add(self, obs: np.ndarray) -> None:
        obs = np.ascontiguousarray(obs)
        self._frame_hashes.append(_hash64(obs.tobytes()))
        if self._channel_hashers is None:
            self._channel_hashers = [_new_hasher() for _ in range(obs.shape[0])]
        assert obs.shape[0] == len(self._channel_hashers), (
            f"channel count changed mid-walk: {obs.shape[0]} vs {len(self._channel_hashers)}"
        )
        for c, h in enumerate(self._channel_hashers):
            h.update(obs[c].tobytes())

    def digest(self) -> ObsDigest:
        assert self._channel_hashers is not None, "no frames added"
        return ObsDigest(
            frame_hashes=np.array(self._frame_hashes, dtype=np.uint64),
            channel_hashes=np.array(
                [_as_uint64(h.digest()) for h in self._channel_hashers], dtype=np.uint64
            ),
        )
