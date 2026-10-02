import numpy as np

from training.goldens.lib.content import ArrayLayout, layout_of
from training.goldens.lib.diff import (
    ArrayAdded,
    ArrayLayoutChanged,
    ArrayRemoved,
    ArrayRowsChanged,
    ContentDiff,
    diff_content,
)
from training.goldens.lib.hashing import row_hashes


def test_rows_compared_by_bytes_not_values() -> None:
    stored = np.array([0.0, np.nan, 1.0])
    got = stored.copy()
    assert diff_content({"a": got}, {"a": stored}) is None   # NaN rows are equal by bytes

    got[0] = -0.0   # equal by value, not by bytes
    assert diff_content({"a": got}, {"a": stored}) == ContentDiff(
        changes=(ArrayRowsChanged(name="a", layout=layout_of(stored), rows=(0,)),)
    )


def test_layout_changes() -> None:
    arr = np.arange(6, dtype=np.int64).reshape(3, 2)

    d = diff_content({"a": arr.astype(np.float64)}, {"a": arr})
    assert d is not None and d.changes == (
        ArrayLayoutChanged(name="a", stored=layout_of(arr), got=layout_of(arr.astype(np.float64))),
    )

    # Switching between full and hashed storage is a layout change.
    hashed = row_hashes(arr)
    d = diff_content({"a": hashed}, {"a": arr})
    assert d is not None and d.changes == (
        ArrayLayoutChanged(name="a", stored=layout_of(arr), got=hashed.layout),
    )

    # A dtype change in hashed data reads as a layout change, not as every row changed.
    d = diff_content({"a": row_hashes(arr.astype(np.int32))}, {"a": hashed})
    assert d is not None and [type(c) for c in d.changes] == [ArrayLayoutChanged]


def test_added_and_removed_arrays() -> None:
    a, b = np.zeros(2, dtype=np.int64), np.ones(2, dtype=np.int64)
    d = diff_content({"c": a, "b": b}, {"b": b, "a": a})
    assert d == ContentDiff(
        changes=(
            ArrayRemoved(name="a", stored=layout_of(a)),
            ArrayAdded(name="c", got=layout_of(a)),
        )
    )


def test_zero_rows() -> None:
    empty = np.zeros((0, 3), dtype=np.float32)
    assert diff_content({"a": empty}, {"a": empty.copy()}) is None

    d = diff_content({"a": np.zeros((2, 3), dtype=np.float32)}, {"a": empty})
    assert d is not None and d.changes == (
        ArrayLayoutChanged(
            name="a",
            stored=ArrayLayout(shape=(0, 3), dtype=np.dtype(np.float32), hashed_axis=None),
            got=ArrayLayout(shape=(2, 3), dtype=np.dtype(np.float32), hashed_axis=None),
        ),
    )
