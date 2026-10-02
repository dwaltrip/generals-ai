from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from training.goldens.lib.content import (
    ArrayLayout,
    EntryContent,
    RowHashes,
    layout_of,
    row_data,
)


@dataclass(frozen=True)
class ArrayAdded:
    name: str
    got: ArrayLayout


@dataclass(frozen=True)
class ArrayRemoved:
    name: str
    stored: ArrayLayout


@dataclass(frozen=True)
class ArrayLayoutChanged:
    name: str
    stored: ArrayLayout
    got: ArrayLayout


@dataclass(frozen=True)
class ArrayRowsChanged:
    name: str
    layout: ArrayLayout       # the same on both sides
    rows: tuple[int, ...]     # indices along axis 0 of row_data


type ArrayChange = ArrayAdded | ArrayRemoved | ArrayLayoutChanged | ArrayRowsChanged


@dataclass(frozen=True)
class ContentDiff:
    changes: tuple[ArrayChange, ...]   # ordered by array name


# Content is equal exactly when this returns None.
def diff_content(got: EntryContent, stored: EntryContent) -> ContentDiff | None:
    changes: list[ArrayChange] = []
    for name in sorted(got.keys() | stored.keys()):
        if name not in stored:
            changes.append(ArrayAdded(name=name, got=layout_of(got[name])))
        elif name not in got:
            changes.append(ArrayRemoved(name=name, stored=layout_of(stored[name])))
        elif (change := _diff_array(name, got[name], stored[name])) is not None:
            changes.append(change)
    return ContentDiff(changes=tuple(changes)) if changes else None


def _diff_array(
    name: str, got: np.ndarray | RowHashes, stored: np.ndarray | RowHashes
) -> ArrayChange | None:
    got_layout, stored_layout = layout_of(got), layout_of(stored)
    if got_layout != stored_layout:
        return ArrayLayoutChanged(name=name, stored=stored_layout, got=got_layout)
    rows = _changed_rows(row_data(got), row_data(stored))
    return ArrayRowsChanged(name=name, layout=got_layout, rows=rows) if rows else None


# Rows are compared by their bytes. Comparing values would differ for floats:
# NaN never equals itself, and -0.0 equals 0.0.
def _changed_rows(got: np.ndarray, stored: np.ndarray) -> tuple[int, ...]:
    # The row width is explicit, since reshape(n, -1) fails when n is 0.
    n_rows = got.shape[0]
    width = got.dtype.itemsize * math.prod(got.shape[1:])
    got_b = np.ascontiguousarray(got).view(np.uint8).reshape(n_rows, width)
    stored_b = np.ascontiguousarray(stored).view(np.uint8).reshape(n_rows, width)
    return tuple(int(i) for i in np.nonzero((got_b != stored_b).any(axis=1))[0])
