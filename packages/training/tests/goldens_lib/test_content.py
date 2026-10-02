import numpy as np
import pytest

from training.goldens.lib.content import ArrayLayout, RowHashes, check_content


def _hashes(n: int, dtype: type = np.uint64) -> np.ndarray:
    return np.zeros(n, dtype=dtype)


def _layout(shape: tuple[int, ...], axis: int | None) -> ArrayLayout:
    return ArrayLayout(shape=shape, dtype=np.dtype(np.float32), hashed_axis=axis)


_ARR = np.zeros(3)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param({"": _ARR}, id="empty name"),
        pytest.param({"a/b": _ARR}, id="slash in name"),
        pytest.param({"__meta__": _ARR}, id="reserved: __meta__"),
        pytest.param({"file": _ARR}, id="reserved: file"),
        pytest.param({"allow_pickle": _ARR}, id="reserved: allow_pickle"),
        pytest.param({"a": [1, 2, 3]}, id="not an array"),
        pytest.param({"a": np.array([None, 1], dtype=object)}, id="object dtype"),
        pytest.param({"a": np.array(1.0)}, id="zero-dimensional"),
        pytest.param({"a": RowHashes(_hashes(3), _layout((3, 2), 2))}, id="axis out of range"),
        pytest.param({"a": RowHashes(_hashes(3), _layout((3, 2), None))}, id="axis missing"),
        pytest.param({"a": RowHashes(_hashes(3, np.int64), _layout((3, 2), 0))}, id="hash dtype"),
        pytest.param({"a": RowHashes(_hashes(4), _layout((3, 2), 0))}, id="hash count"),
    ],
)
def test_bad_content_rejected(content: dict) -> None:
    with pytest.raises(ValueError):
        check_content(content)
