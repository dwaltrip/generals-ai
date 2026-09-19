import numpy as np

from training.goldens.compare import diff_rows


def test_diff_rows_compares_bytes_not_values() -> None:
    ref = np.array([0.0, np.nan, 1.0])
    got = ref.copy()
    assert diff_rows("k", got, ref) is None   # NaN rows are equal by bytes

    got[0] = -0.0   # equal by value, not by bytes
    d = diff_rows("k", got, ref)
    assert d is not None and d.changed_rows is not None
    assert d.changed_rows.tolist() == [0]


def test_diff_rows_layout_mismatch() -> None:
    ref = np.zeros(3, dtype=np.int64)
    d = diff_rows("k", ref.astype(np.float64), ref)
    assert d is not None and d.changed_rows is None
    assert d.ref.dtype == np.int64 and d.got.dtype == np.float64
