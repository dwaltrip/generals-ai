import rust_py_demo
from rust_py_demo import _native

print("=== via top-level (Python wrapper) ===")
print("add(2, 3) =", rust_py_demo.add(2, 3))
print("sum_squares([1,2,3,4,5]) =", rust_py_demo.sum_squares([1, 2, 3, 4, 5]))
print("mean_of_squares([1,2,3,4,5]) =", rust_py_demo.mean_of_squares([1, 2, 3, 4, 5]))

print()
print("=== via _native (direct Rust) ===")
print("_native.add(2, 3) =", _native.add(2, 3))
