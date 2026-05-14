use std::collections::{HashMap, VecDeque};

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct DeathEvent {
    pub timestep: i32,
    pub player: usize,
}

#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct CaptureEvent {
    pub timestep: i32,
    pub captor: usize,
    pub captured: usize,
}

#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct NeutralizeEvent {
    pub timestep: i32,
    pub player: usize,
}

#[pyclass]
pub struct State {
    // Grids — flat row-major [H*W]
    pub ownership: Vec<i8>,
    pub armies: Vec<i32>,
    pub cities_mask: Vec<u8>, // numpy bool ⇆ u8

    // Structures
    pub cities: Vec<i32>,
    pub generals: Vec<i32>, // -1 sentinel for captured/missing

    // Per-player flags + queues
    pub alive: Vec<bool>,
    pub has_kill: Vec<bool>,
    pub input_buffer: Vec<VecDeque<usize>>,

    // Game-level scalars
    pub timestep: i32,
    pub num_players: usize,
    pub alive_count: usize,
    pub updates_since_move: i32,
    pub afks_cursor: usize,
    pub moves_cursor: usize,

    // Damage matrices, P*P row-major
    pub damage_sym_all: Vec<i32>,
    pub damage_sym_pre: Vec<i32>,
    pub damage_off_all: Vec<i32>,
    pub damage_off_pre: Vec<i32>,

    // Event lists
    pub death_events: Vec<DeathEvent>,
    pub capture_events: Vec<CaptureEvent>,
    pub neutralize_events: Vec<NeutralizeEvent>,

    // Per-perspective output (curated players only)
    pub perspective_indices: HashMap<usize, usize>,
    pub actions_source: Vec<i16>, // K * T_max flat
    pub actions_dest: Vec<i16>,
    pub actions_is50: Vec<u8>,
    pub k: usize,
    pub t_max: usize,

    // Snapshots — per-tick frozen grids, narrowed to int16 for armies
    pub snapshots_ownership: Vec<Vec<i8>>,
    pub snapshots_armies: Vec<Vec<i16>>,
    pub snapshots_cities_mask: Vec<Vec<u8>>,

    // Cached
    pub map_size: usize,
}

#[pymethods]
impl State {
    /// Construct a Rust State from a Python `replay_parser.state.State`. Used by
    /// parity tests to mirror the Python sim's state at any timestep.
    #[classmethod]
    #[pyo3(name = "from_python")]
    fn py_from_python(_cls: &Bound<'_, pyo3::types::PyType>, py_state: &Bound<'_, PyAny>) -> PyResult<Self> {
        let ownership: PyReadonlyArray1<i8> = py_state.getattr("ownership")?.extract()?;
        let armies: PyReadonlyArray1<i32> = py_state.getattr("armies")?.extract()?;
        let cities_mask_arr: PyReadonlyArray1<bool> = py_state.getattr("cities_mask")?.extract()?;
        let damage_sym_all = py_state.getattr("damage_sym_all")?;
        let damage_sym_pre = py_state.getattr("damage_sym_pre")?;
        let damage_off_all = py_state.getattr("damage_off_all")?;
        let damage_off_pre = py_state.getattr("damage_off_pre")?;
        let actions_source = py_state.getattr("actions_source")?;
        let actions_dest = py_state.getattr("actions_dest")?;
        let actions_is50 = py_state.getattr("actions_is50")?;

        let cities_mask: Vec<u8> = cities_mask_arr.as_slice()?.iter().map(|&b| b as u8).collect();
        let map_size = ownership.len()?;

        let cities: Vec<i32> = py_state.getattr("cities")?.extract()?;
        let generals: Vec<i32> = py_state.getattr("generals")?.extract()?;
        let alive: Vec<bool> = py_state.getattr("alive")?.extract()?;
        let has_kill: Vec<bool> = py_state.getattr("has_kill")?.extract()?;

        // input_buffer is list[deque[int]]. PyO3 extracts each deque as a list (iterable).
        let raw_buffers: Vec<Vec<usize>> = py_state.getattr("input_buffer")?.extract()?;
        let input_buffer: Vec<VecDeque<usize>> =
            raw_buffers.into_iter().map(VecDeque::from).collect();

        let perspective_indices: HashMap<usize, usize> =
            py_state.getattr("perspective_indices")?.extract()?;

        // Damage matrices flatten via numpy ravel (we keep flat in Rust).
        let damage_sym_all = flatten_2d_i32(damage_sym_all)?;
        let damage_sym_pre = flatten_2d_i32(damage_sym_pre)?;
        let damage_off_all = flatten_2d_i32(damage_off_all)?;
        let damage_off_pre = flatten_2d_i32(damage_off_pre)?;
        let (actions_source, k, t_max) = flatten_2d_i16(actions_source)?;
        let (actions_dest, _, _) = flatten_2d_i16(actions_dest)?;
        let (actions_is50, _, _) = flatten_2d_u8(actions_is50)?;

        let death_events = extract_events_2(py_state.getattr("death_events")?, "player")?
            .into_iter()
            .map(|(timestep, player)| DeathEvent { timestep, player })
            .collect();
        let neutralize_events =
            extract_events_2(py_state.getattr("neutralize_events")?, "player")?
                .into_iter()
                .map(|(timestep, player)| NeutralizeEvent { timestep, player })
                .collect();
        let capture_events = extract_capture_events(py_state.getattr("capture_events")?)?;

        let timestep: i32 = py_state.getattr("timestep")?.extract()?;
        let num_players: usize = py_state.getattr("num_players")?.extract()?;
        let alive_count: usize = py_state.getattr("alive_count")?.extract()?;
        let updates_since_move: i32 = py_state.getattr("updates_since_move")?.extract()?;
        let afks_cursor: usize = py_state.getattr("afks_cursor")?.extract()?;
        let moves_cursor: usize = py_state.getattr("moves_cursor")?.extract()?;

        Ok(State {
            ownership: ownership.as_slice()?.to_vec(),
            armies: armies.as_slice()?.to_vec(),
            cities_mask,
            cities,
            generals,
            alive,
            has_kill,
            input_buffer,
            timestep,
            num_players,
            alive_count,
            updates_since_move,
            afks_cursor,
            moves_cursor,
            damage_sym_all,
            damage_sym_pre,
            damage_off_all,
            damage_off_pre,
            death_events,
            capture_events,
            neutralize_events,
            perspective_indices,
            actions_source,
            actions_dest,
            actions_is50,
            k,
            t_max,
            snapshots_ownership: Vec::new(),
            snapshots_armies: Vec::new(),
            snapshots_cities_mask: Vec::new(),
            map_size,
        })
    }

    // --- Output getters (numpy) ---
    #[getter]
    fn timestep(&self) -> i32 {
        self.timestep
    }
    #[getter]
    fn alive_count(&self) -> usize {
        self.alive_count
    }
    #[getter]
    fn moves_cursor(&self) -> usize {
        self.moves_cursor
    }
    #[getter]
    fn updates_since_move(&self) -> i32 {
        self.updates_since_move
    }

    #[getter]
    fn ownership<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i8>> {
        self.ownership.clone().into_pyarray(py)
    }
    #[getter]
    fn armies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.armies.clone().into_pyarray(py)
    }
    #[getter]
    fn cities_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        self.cities_mask.clone().into_pyarray(py)
    }
    #[getter]
    fn actions_source<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i16>> {
        self.actions_source.clone().into_pyarray(py)
    }
    #[getter]
    fn actions_dest<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i16>> {
        self.actions_dest.clone().into_pyarray(py)
    }
    #[getter]
    fn actions_is50<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        self.actions_is50.clone().into_pyarray(py)
    }
    #[getter]
    fn input_buffer_lengths(&self) -> Vec<usize> {
        self.input_buffer.iter().map(|d| d.len()).collect()
    }
    #[getter]
    fn input_buffer_contents(&self) -> Vec<Vec<usize>> {
        self.input_buffer
            .iter()
            .map(|d| d.iter().copied().collect())
            .collect()
    }

    // --- Step body methods (B1: easy ones; rest stubbed) ---

    /// Mirror of replay_parser.step.apply_production (state.timestep already advanced).
    fn apply_production(&mut self) {
        if self.timestep % 2 == 0 {
            for &g in &self.generals {
                if g >= 0 {
                    self.armies[g as usize] += 1;
                }
            }
            for &c in &self.cities {
                if self.ownership[c as usize] >= 0 {
                    self.armies[c as usize] += 1;
                }
            }
        }
        if self.timestep % 50 == 0 {
            for i in 0..self.map_size {
                if self.ownership[i] >= 0 {
                    self.armies[i] += 1;
                }
            }
        }
    }

    /// Mirror of replay_parser.step.buffer_pending_moves.
    #[pyo3(name = "buffer_pending_moves")]
    fn py_buffer_pending_moves(
        &mut self,
        m_timestep: PyReadonlyArray1<i32>,
        m_index: PyReadonlyArray1<i8>,
    ) -> PyResult<()> {
        let ts = m_timestep.as_slice()?;
        let idx = m_index.as_slice()?;
        while self.moves_cursor < ts.len() && ts[self.moves_cursor] <= self.timestep {
            let p = idx[self.moves_cursor] as usize;
            self.input_buffer[p].push_back(self.moves_cursor);
            self.moves_cursor += 1;
        }
        Ok(())
    }

    /// Mirror of replay_parser.step.select_candidates. Returns selected move indices.
    #[pyo3(name = "select_candidates")]
    fn py_select_candidates(
        &mut self,
        m_index: PyReadonlyArray1<i8>,
        m_source: PyReadonlyArray1<i16>,
        m_dest: PyReadonlyArray1<i16>,
    ) -> PyResult<Vec<usize>> {
        let m_idx = m_index.as_slice()?;
        let m_src = m_source.as_slice()?;
        let m_dst = m_dest.as_slice()?;
        let mut candidates: Vec<usize> = Vec::new();
        for p in 0..self.num_players {
            while let Some(&i) = self.input_buffer[p].front() {
                self.input_buffer[p].pop_front();
                if self.is_valid(i, m_idx, m_src, m_dst) {
                    candidates.push(i);
                    break;
                }
            }
        }
        Ok(candidates)
    }

    /// Mirror of replay_parser.moves.is_valid.
    #[pyo3(name = "is_valid")]
    fn py_is_valid(
        &self,
        move_idx: usize,
        m_index: PyReadonlyArray1<i8>,
        m_source: PyReadonlyArray1<i16>,
        m_dest: PyReadonlyArray1<i16>,
    ) -> PyResult<bool> {
        Ok(self.is_valid(
            move_idx,
            m_index.as_slice()?,
            m_source.as_slice()?,
            m_dest.as_slice()?,
        ))
    }

    /// Mirror of replay_parser.moves.record_action.
    #[pyo3(name = "record_action")]
    fn py_record_action(
        &mut self,
        move_idx: usize,
        m_index: PyReadonlyArray1<i8>,
        m_source: PyReadonlyArray1<i16>,
        m_dest: PyReadonlyArray1<i16>,
        m_is50: PyReadonlyArray1<u8>,
    ) -> PyResult<()> {
        let p = m_index.as_slice()?[move_idx] as usize;
        let Some(&ps) = self.perspective_indices.get(&p) else {
            return Ok(());
        };
        let t = self.timestep as usize;
        let row = ps * self.t_max + t;
        self.actions_source[row] = m_source.as_slice()?[move_idx];
        self.actions_dest[row] = m_dest.as_slice()?[move_idx];
        self.actions_is50[row] = m_is50.as_slice()?[move_idx];
        Ok(())
    }
}

// Inner (non-PyO3) helpers — same names as the #[pymethods] dispatchers.
impl State {
    fn is_valid(&self, move_idx: usize, m_index: &[i8], m_source: &[i16], m_dest: &[i16]) -> bool {
        let source = m_source[move_idx] as usize;
        let dest = m_dest[move_idx] as usize;
        let mover = m_index[move_idx] as i8;
        self.ownership[source] == mover && self.ownership[dest] != -2 && self.armies[source] >= 2
    }
}

// --- Helpers for from_python ---

fn flatten_2d_i32(arr: Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
    let ravel: Vec<i32> = arr.call_method0("ravel")?.extract()?;
    Ok(ravel)
}

fn flatten_2d_i16(arr: Bound<'_, PyAny>) -> PyResult<(Vec<i16>, usize, usize)> {
    let shape: (usize, usize) = arr.getattr("shape")?.extract()?;
    let ravel: Vec<i16> = arr.call_method0("ravel")?.extract()?;
    Ok((ravel, shape.0, shape.1))
}

fn flatten_2d_u8(arr: Bound<'_, PyAny>) -> PyResult<(Vec<u8>, usize, usize)> {
    let shape: (usize, usize) = arr.getattr("shape")?.extract()?;
    let ravel: Vec<u8> = arr.call_method0("ravel")?.extract()?;
    Ok((ravel, shape.0, shape.1))
}

fn extract_events_2(events: Bound<'_, PyAny>, second_field: &str) -> PyResult<Vec<(i32, usize)>> {
    let mut out = Vec::new();
    for e in events.try_iter()? {
        let e = e?;
        let t: i32 = e.getattr("timestep")?.extract()?;
        let p: usize = e.getattr(second_field)?.extract()?;
        out.push((t, p));
    }
    Ok(out)
}

fn extract_capture_events(events: Bound<'_, PyAny>) -> PyResult<Vec<CaptureEvent>> {
    let mut out = Vec::new();
    for e in events.try_iter()? {
        let e = e?;
        let timestep: i32 = e.getattr("timestep")?.extract()?;
        let captor: usize = e.getattr("captor")?.extract()?;
        let captured: usize = e.getattr("captured")?.extract()?;
        out.push(CaptureEvent {
            timestep,
            captor,
            captured,
        });
    }
    Ok(out)
}
