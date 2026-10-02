import numpy as np
import pytest

from training.goldens.lib.hashing import StreamingRowHasher, row_hashes


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.uint8])
def test_streaming_matches_row_hashes(dtype: type) -> None:
    rng = np.random.default_rng(0)
    arr = (rng.random((5, 3, 4, 2)) * 100).astype(dtype)
    axes = (0, 1, 3)

    hasher = StreamingRowHasher(axes)
    for row in arr:
        hasher.add(row)

    for got, axis in zip(hasher.hashes(), axes, strict=True):
        expected = row_hashes(arr, axis)
        assert got.layout == expected.layout
        assert np.array_equal(got.hashes, expected.hashes)
        # Memory order doesn't affect the hashes.
        assert np.array_equal(row_hashes(np.asfortranarray(arr), axis).hashes, expected.hashes)


def test_streaming_rejects_layout_change() -> None:
    hasher = StreamingRowHasher((0, 1))
    with pytest.raises(RuntimeError):
        hasher.hashes()
    hasher.add(np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        hasher.add(np.zeros((3, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        hasher.add(np.zeros((3, 4), dtype=np.float16))


def test_streaming_one_dimensional() -> None:
    arr = np.arange(5, dtype=np.int32)
    hasher = StreamingRowHasher((0,))
    for i in range(arr.shape[0]):
        hasher.add(arr[i, ...])   # 0-d slices

    (got,) = hasher.hashes()
    expected = row_hashes(arr)
    assert got.layout == expected.layout
    assert np.array_equal(got.hashes, expected.hashes)
