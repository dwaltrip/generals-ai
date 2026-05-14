use pyo3::prelude::*;

#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[pyfunction]
fn sum_squares(values: Vec<i64>) -> i64 {
    values.iter().map(|v| v * v).sum()
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sum_squares, m)?)?;
    Ok(())
}
