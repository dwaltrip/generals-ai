from collections.abc import Sequence

from rust_py_demo._native import add, sum_squares


def mean_of_squares(values: Sequence[int]) -> float:
    """Pure-Python wrapper that builds on the Rust `sum_squares`."""
    if not values:
        raise ValueError("values must be non-empty")
    return sum_squares(values) / len(values)


__all__ = ["add", "sum_squares", "mean_of_squares"]
