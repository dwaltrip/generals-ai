use pyo3::prelude::*;

mod state;

use state::{CaptureEvent, DeathEvent, NeutralizeEvent, State};

#[pyfunction]
fn ping() {}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_class::<State>()?;
    m.add_class::<DeathEvent>()?;
    m.add_class::<CaptureEvent>()?;
    m.add_class::<NeutralizeEvent>()?;
    Ok(())
}
