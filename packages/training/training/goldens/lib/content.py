from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class ArrayLayout:
    # The shape and dtype are from the raw array, even when hashed
    shape: tuple[int, ...]
    dtype: np.dtype
    # None: stored in full (no hashing)
    hashed_axis: int | None


@dataclass(frozen=True)
class RowHashes:
    hashes: np.ndarray        # uint64, one per index along layout.hashed_axis
    layout: ArrayLayout


type EntryContent = dict[str, np.ndarray | RowHashes]


def layout_of(value: np.ndarray | RowHashes) -> ArrayLayout:
    if isinstance(value, RowHashes):
        return value.layout
    return ArrayLayout(shape=value.shape, dtype=value.dtype, hashed_axis=None)


# The array that is compared row by row.
def row_data(value: np.ndarray | RowHashes) -> np.ndarray:
    return value.hashes if isinstance(value, RowHashes) else value


def content_id(value: np.ndarray | RowHashes) -> str:
    layout = layout_of(value)
    h = hashlib.sha256()
    h.update(repr((layout.shape, layout.dtype.str, layout.hashed_axis)).encode())
    h.update(np.ascontiguousarray(row_data(value)).tobytes())
    return h.hexdigest()[:16]


# `__meta__` is used by the store. `file` and `allow_pickle` are keyword
# parameters of `np.savez`.
_RESERVED_NAMES = frozenset({"__meta__", "file", "allow_pickle"})


def check_content(content: EntryContent) -> None:
    for name, value in content.items():
        if not (isinstance(name, str) and name and "/" not in name):
            raise ValueError(f"bad array name: {name!r}")
        if name in _RESERVED_NAMES:
            raise ValueError(f"reserved array name: {name!r}")

        if isinstance(value, RowHashes):
            _check_row_hashes(name, value)
        elif isinstance(value, np.ndarray):
            _check_layout(name, layout_of(value))
        else:
            raise ValueError(f"{name}: expected ndarray or RowHashes, got {type(value).__name__}")


def _check_layout(name: str, layout: ArrayLayout) -> None:
    # Object arrays are pickled on save, so their bytes aren't a stable encoding.
    if layout.dtype.hasobject:
        raise ValueError(f"{name}: object dtype {layout.dtype}")
    if len(layout.shape) == 0:
        raise ValueError(f"{name}: zero-dimensional array")


def _check_row_hashes(name: str, value: RowHashes) -> None:
    layout, hashes = value.layout, value.hashes
    _check_layout(name, layout)

    axis = layout.hashed_axis
    if not (isinstance(axis, int) and 0 <= axis < len(layout.shape)):
        raise ValueError(f"{name}: hashed axis {axis} for raw shape {layout.shape}")
    if hashes.dtype != np.uint64 or hashes.shape != (layout.shape[axis],):
        raise ValueError(
            f"{name}: {hashes.dtype} hashes of shape {hashes.shape},"
            f" for raw shape {layout.shape} and axis {axis}"
        )
