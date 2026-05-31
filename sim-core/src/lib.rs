use pyo3::prelude::*;

mod state;

use state::{army_overflow_pyerr, CaptureEvent, DeathEvent, NeutralizeEvent, State};

/// Rust-side mirror of `game_types.StaticMap`. The derived `FromPyObject`
/// defaults to attribute access, so any object exposing these attributes —
/// a `StaticMap`, a `GameStatic.map`, or a test mock — extracts cleanly,
/// preserving the duck-typed boundary.
#[derive(FromPyObject)]
struct StaticMapInput {
    map_width: usize,
    map_height: usize,
    usernames: Vec<String>,
    mountains: Vec<i32>,
    initial_cities: Vec<i32>,
    initial_city_armies: Vec<i32>,
    initial_generals: Vec<i32>,
    initial_neutrals: Vec<i32>,
    initial_neutral_armies: Vec<i32>,
}

/// Build a fresh `State` from a duck-typed static-map record (`StaticMap`).
///
/// Extracts the map's attributes into a `StaticMapInput`, calls
/// `State::build_initial`, and writes the initial pre-loop snapshot so
/// snapshot indexing is consistent across `simulate` and `step_tick`
/// driven loops. `new_state` is handed the `StaticMap` directly; `simulate`
/// reaches it via `replay.static.map`.
fn build_state_from_static(py: Python<'_>, map_data: &Bound<'_, PyAny>) -> PyResult<State> {
    let m: StaticMapInput = map_data.extract()?;
    let map_size = m.map_width * m.map_height;
    let num_players = m.usernames.len();

    let mut state = State::build_initial(
        map_size,
        num_players,
        &m.mountains,
        &m.initial_cities,
        &m.initial_city_armies,
        &m.initial_generals,
        &m.initial_neutrals,
        &m.initial_neutral_armies,
    );
    // Initial snapshot — mirrors the pre-loop append in the old Python parser.
    state.snapshot().map_err(|e| army_overflow_pyerr(py, e))?;
    Ok(state)
}

/// Build a fresh `State` ready for stepwise driving via `State.step_tick`.
/// Companion to `simulate`: same construction, just no moves array.
#[pyfunction]
fn new_state(py: Python<'_>, map_data: &Bound<'_, PyAny>) -> PyResult<State> {
    build_state_from_static(py, map_data)
}

/// Run the full simulator on a Python `ReplayData`. Returns a finished
/// Rust State after running to game-over.
#[pyfunction]
fn simulate(py: Python<'_>, replay: &Bound<'_, PyAny>) -> PyResult<State> {
    use numpy::PyReadonlyArray1;

    let map_data = replay.getattr("static")?.getattr("map")?;
    let mut state = build_state_from_static(py, &map_data)?;

    let moves = replay.getattr("moves")?;
    let m_timestep: PyReadonlyArray1<i32> = moves.getattr("timestep")?.extract()?;
    let m_index: PyReadonlyArray1<i8> = moves.getattr("index")?.extract()?;
    let m_source: PyReadonlyArray1<i16> = moves.getattr("source")?.extract()?;
    let m_dest: PyReadonlyArray1<i16> = moves.getattr("dest")?.extract()?;
    let m_is50: PyReadonlyArray1<u8> = moves.getattr("is50")?.extract()?;

    let afks = replay.getattr("afks")?;
    let a_timestep: PyReadonlyArray1<i32> = afks.getattr("timestep")?.extract()?;
    let a_index: PyReadonlyArray1<i8> = afks.getattr("index")?.extract()?;

    let m_ts = m_timestep.as_slice()?;
    let m_idx = m_index.as_slice()?;
    let m_src = m_source.as_slice()?;
    let m_dst = m_dest.as_slice()?;
    let m_i5 = m_is50.as_slice()?;
    let a_ts = a_timestep.as_slice()?;
    let a_idx = a_index.as_slice()?;

    let result = py.detach(|| -> Result<(), state::ArmyOverflow> {
        loop {
            if !state.step(m_ts, m_idx, m_src, m_dst, m_i5, a_ts, a_idx)? {
                break;
            }
        }
        Ok(())
    });
    result.map_err(|e| army_overflow_pyerr(py, e))?;

    Ok(state)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(new_state, m)?)?;
    m.add_class::<State>()?;
    m.add_class::<DeathEvent>()?;
    m.add_class::<CaptureEvent>()?;
    m.add_class::<NeutralizeEvent>()?;
    Ok(())
}
