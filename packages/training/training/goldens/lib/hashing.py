from __future__ import annotations

import hashlib

import numpy as np

from training.goldens.lib.content import ArrayLayout, RowHashes


# The hash is sha256 truncated to 8 bytes, read as a little-endian uint64, over
# the slice's C-order bytes. Chosen over blake2b for speed (hardware SHA on the M1).


def _new_hasher():
    return hashlib.sha256()


def _as_uint64(digest: bytes) -> int:
    return int.from_bytes(digest[:8], "little")


def _hash64(buf: bytes) -> int:
    h = _new_hasher()
    h.update(buf)
    return _as_uint64(h.digest())


def row_hashes(arr: np.ndarray, axis: int = 0) -> RowHashes:
    if not 0 <= axis < arr.ndim:
        raise ValueError(f"axis {axis} for shape {arr.shape}")

    rows = np.ascontiguousarray(np.moveaxis(arr, axis, 0))
    hashes = np.array([_hash64(rows[i].tobytes()) for i in range(rows.shape[0])], dtype=np.uint64)
    return RowHashes(
        hashes=hashes,
        layout=ArrayLayout(shape=arr.shape, dtype=arr.dtype, hashed_axis=axis),
    )


# This exists solely for reduced memory use when hashing large arrays.
# Hashes over the array one slice at a time, along axis 0.
# Produces hashes identical to those from `row_hashes`.
class StreamingRowHasher:
    def __init__(self, axes: tuple[int, ...]) -> None:
        if not (axes and len(set(axes)) == len(axes) and min(axes) >= 0):
            raise ValueError(f"bad axes: {axes}")

        self._axes = axes
        self._slice_shape: tuple[int, ...] | None = None
        self._dtype: np.dtype | None = None
        self._n_slices = 0
        self._slice_hashes: list[int] = []
        # For each requested axis other than 0: one incremental hasher per index.
        self._index_hashers: dict[int, list] = {}

    def add(self, row: np.ndarray) -> None:
        row = np.asarray(row)
        if self._slice_shape is None:
            if max(self._axes) > row.ndim:
                raise ValueError(f"axes {self._axes} for slice shape {row.shape}")
            self._slice_shape, self._dtype = row.shape, row.dtype
            self._index_hashers = {
                axis: [_new_hasher() for _ in range(row.shape[axis - 1])]
                for axis in self._axes
                if axis > 0
            }
        if row.shape != self._slice_shape or row.dtype != self._dtype:
            raise ValueError(
                f"slice layout changed: {row.shape} {row.dtype}"
                f" vs {self._slice_shape} {self._dtype}"
            )

        self._n_slices += 1
        if 0 in self._axes:
            self._slice_hashes.append(_hash64(row.tobytes()))
        for axis, hashers in self._index_hashers.items():
            parts = np.ascontiguousarray(np.moveaxis(row, axis - 1, 0))
            for i, h in enumerate(hashers):
                h.update(parts[i].tobytes())

    def hashes(self) -> tuple[RowHashes, ...]:
        if self._slice_shape is None or self._dtype is None:
            raise RuntimeError("no slices added")

        shape = (self._n_slices, *self._slice_shape)
        out = []
        for axis in self._axes:
            if axis == 0:
                values = self._slice_hashes
            else:
                values = [_as_uint64(h.digest()) for h in self._index_hashers[axis]]
            out.append(
                RowHashes(
                    hashes=np.array(values, dtype=np.uint64),
                    layout=ArrayLayout(shape=shape, dtype=self._dtype, hashed_axis=axis),
                )
            )
        return tuple(out)
