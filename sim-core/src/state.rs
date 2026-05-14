use std::collections::VecDeque;

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

const ARMY_MAX: i32 = 32767;
const MAX_ALL_AFK_TIMESTEPS: i32 = 2000;
const MAX_GAME_TIMESTEPS: i32 = 50000;

#[pyclass]
#[derive(Clone, Debug)]
pub struct DeathEvent {
    pub timestep: i32,
    pub player: usize,
}

#[pymethods]
impl DeathEvent {
    #[getter]
    fn timestep(&self) -> i32 {
        self.timestep
    }
    #[getter]
    fn player(&self) -> usize {
        self.player
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct CaptureEvent {
    pub timestep: i32,
    pub captor: usize,
    pub captured: usize,
}

#[pymethods]
impl CaptureEvent {
    #[getter]
    fn timestep(&self) -> i32 {
        self.timestep
    }
    #[getter]
    fn captor(&self) -> usize {
        self.captor
    }
    #[getter]
    fn captured(&self) -> usize {
        self.captured
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct NeutralizeEvent {
    pub timestep: i32,
    pub player: usize,
}

#[pymethods]
impl NeutralizeEvent {
    #[getter]
    fn timestep(&self) -> i32 {
        self.timestep
    }
    #[getter]
    fn player(&self) -> usize {
        self.player
    }
}

#[derive(Debug)]
pub struct ArmyOverflow {
    pub timestep: i32,
    pub value: i32,
}

#[pyclass]
pub struct State {
    pub ownership: Vec<i8>,
    pub armies: Vec<i32>,
    pub cities_mask: Vec<u8>,

    pub cities: Vec<i32>,
    pub generals: Vec<i32>,

    pub alive: Vec<bool>,
    pub has_kill: Vec<bool>,
    pub input_buffer: Vec<VecDeque<usize>>,

    pub timestep: i32,
    pub num_players: usize,
    pub alive_count: usize,
    pub updates_since_move: i32,
    pub afks_cursor: usize,
    pub moves_cursor: usize,

    pub damage_off_all: Vec<i32>,

    pub death_events: Vec<DeathEvent>,
    pub capture_events: Vec<CaptureEvent>,
    pub neutralize_events: Vec<NeutralizeEvent>,

    pub snapshots_ownership: Vec<Vec<i8>>,
    pub snapshots_armies: Vec<Vec<i16>>,
    pub snapshots_cities_mask: Vec<Vec<u8>>,

    pub map_size: usize,
}

// ============================================================================
// Python surface (#[pymethods]) — getters only. Inner sim logic in the plain
// `impl State` block below; the hot path runs without GIL.
// ============================================================================

#[pymethods]
impl State {
    #[getter]
    fn timestep(&self) -> i32 {
        self.timestep
    }
    #[getter]
    fn alive_count(&self) -> usize {
        self.alive_count
    }
    #[getter]
    fn num_players(&self) -> usize {
        self.num_players
    }
    #[getter]
    fn alive(&self) -> Vec<bool> {
        self.alive.clone()
    }
    #[getter]
    fn has_kill(&self) -> Vec<bool> {
        self.has_kill.clone()
    }
    #[getter]
    fn generals(&self) -> Vec<i32> {
        self.generals.clone()
    }
    #[getter]
    fn cities(&self) -> Vec<i32> {
        self.cities.clone()
    }
    #[getter]
    fn death_events(&self) -> Vec<DeathEvent> {
        self.death_events.clone()
    }
    #[getter]
    fn capture_events(&self) -> Vec<CaptureEvent> {
        self.capture_events.clone()
    }
    #[getter]
    fn neutralize_events(&self) -> Vec<NeutralizeEvent> {
        self.neutralize_events.clone()
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
    fn damage_off_all<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i32>>> {
        damage_to_pyarray2(py, &self.damage_off_all, self.num_players)
    }

    #[getter]
    fn snapshots_ownership<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<i8>>> {
        self.snapshots_ownership
            .iter()
            .map(|v| v.clone().into_pyarray(py))
            .collect()
    }
    #[getter]
    fn snapshots_armies<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<i16>>> {
        self.snapshots_armies
            .iter()
            .map(|v| v.clone().into_pyarray(py))
            .collect()
    }
    #[getter]
    fn snapshots_cities_mask<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<u8>>> {
        self.snapshots_cities_mask
            .iter()
            .map(|v| v.clone().into_pyarray(py))
            .collect()
    }
    #[getter]
    fn snapshots_len(&self) -> usize {
        self.snapshots_ownership.len()
    }
}

// ============================================================================
// Inner methods (no PyO3, no GIL needed) — the actual sim logic
// ============================================================================

impl State {
    pub(crate) fn apply_production_impl(&mut self) {
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

    pub(crate) fn buffer_pending_moves(&mut self, m_timestep: &[i32], m_index: &[i8]) {
        while self.moves_cursor < m_timestep.len()
            && m_timestep[self.moves_cursor] <= self.timestep
        {
            let p = m_index[self.moves_cursor] as usize;
            self.input_buffer[p].push_back(self.moves_cursor);
            self.moves_cursor += 1;
        }
    }

    pub(crate) fn select_candidates(
        &mut self,
        m_index: &[i8],
        m_source: &[i16],
        m_dest: &[i16],
    ) -> Vec<usize> {
        let mut candidates: Vec<usize> = Vec::new();
        for p in 0..self.num_players {
            while let Some(&i) = self.input_buffer[p].front() {
                self.input_buffer[p].pop_front();
                if self.is_valid(i, m_index, m_source, m_dest) {
                    candidates.push(i);
                    break;
                }
            }
        }
        candidates
    }

    pub(crate) fn is_valid(
        &self,
        move_idx: usize,
        m_index: &[i8],
        m_source: &[i16],
        m_dest: &[i16],
    ) -> bool {
        let source = m_source[move_idx] as usize;
        let dest = m_dest[move_idx] as usize;
        let mover = m_index[move_idx];
        self.ownership[source] == mover
            && self.ownership[dest] != -2
            && self.armies[source] >= 2
    }

    // --- Combat (combat.py mirror) ---

    pub(crate) fn attack(
        &mut self,
        move_idx: usize,
        m_index: &[i8],
        m_source: &[i16],
        m_dest: &[i16],
        m_is50: &[u8],
    ) {
        let source = m_source[move_idx] as usize;
        let dest = m_dest[move_idx] as usize;
        let mover = m_index[move_idx];
        let is50 = m_is50[move_idx] != 0;

        let src_army = self.armies[source];
        let move_reserve = if is50 { (src_army + 1) / 2 } else { 1 };
        let incoming = src_army - move_reserve;

        self.armies[source] = src_army - incoming;

        let dest_owner = self.ownership[dest];
        let dest_army = self.armies[dest];

        let mut damage: i32 = 0;
        if dest_owner == mover {
            self.armies[dest] = dest_army + incoming;
        } else if dest_army >= incoming {
            self.armies[dest] = dest_army - incoming;
            damage = incoming;
        } else {
            self.armies[dest] = incoming - dest_army;
            self.ownership[dest] = mover;
            damage = dest_army;
        }

        if damage > 0 && dest_owner >= 0 {
            let m = mover as usize;
            let d = dest_owner as usize;
            let p = self.num_players;
            self.damage_off_all[m * p + d] += damage;
        }
    }

    pub(crate) fn execute_attack(
        &mut self,
        move_idx: usize,
        m_index: &[i8],
        m_source: &[i16],
        m_dest: &[i16],
        m_is50: &[u8],
    ) {
        let dest = m_dest[move_idx] as usize;
        let old_owner = self.ownership[dest];

        self.attack(move_idx, m_index, m_source, m_dest, m_is50);

        let new_owner = self.ownership[dest];
        if old_owner != new_owner
            && old_owner >= 0
            && self.generals[old_owner as usize] == dest as i32
        {
            self.execute_player_capture(old_owner as usize, new_owner as usize);
        }
    }

    pub(crate) fn execute_player_capture(&mut self, captured: usize, captor: usize) {
        let general_tile = self.generals[captured] as usize;

        // Combat already flipped general_tile to captor; mask flip + halving
        // touches captured player's *other* tiles.
        for i in 0..self.map_size {
            if self.ownership[i] == captured as i8 {
                self.ownership[i] = captor as i8;
                self.armies[i] = (self.armies[i] + 1) / 2;
            }
        }

        self.has_kill[captor] = true;

        if self.alive[captured] {
            self.kill_player(captured);
        }

        self.cities.push(general_tile as i32);
        self.cities_mask[general_tile] = 1;
        self.generals[captured] = -1;

        self.capture_events.push(CaptureEvent {
            timestep: self.timestep,
            captor,
            captured,
        });
    }

    pub(crate) fn try_neutralize_player(&mut self, p: usize) {
        if self.generals[p] < 0 {
            return;
        }
        let general_tile = self.generals[p] as usize;

        for i in 0..self.map_size {
            if self.ownership[i] == p as i8 {
                self.ownership[i] = -1;
            }
        }

        self.cities.push(general_tile as i32);
        self.cities_mask[general_tile] = 1;
        self.generals[p] = -1;
        self.neutralize_events.push(NeutralizeEvent {
            timestep: self.timestep,
            player: p,
        });
    }

    pub(crate) fn kill_player(&mut self, p: usize) {
        self.alive[p] = false;
        self.alive_count -= 1;
        self.death_events.push(DeathEvent {
            timestep: self.timestep,
            player: p,
        });
        self.input_buffer[p].clear();
    }

    pub(crate) fn kill_all_but_strongest_impl(&mut self) {
        let living: Vec<usize> = (0..self.num_players).filter(|&p| self.alive[p]).collect();
        if living.len() <= 1 {
            return;
        }
        let mut armies_per_p = vec![0i64; self.num_players];
        let mut tiles_per_p = vec![0i64; self.num_players];
        for i in 0..self.map_size {
            let o = self.ownership[i];
            if o >= 0 {
                armies_per_p[o as usize] += self.armies[i] as i64;
                tiles_per_p[o as usize] += 1;
            }
        }
        let mut sorted = living.clone();
        sorted.sort_by_key(|&p| (armies_per_p[p], tiles_per_p[p], p as i64));
        // Kill all but the last (strongest)
        for &p in &sorted[..sorted.len() - 1] {
            self.kill_player(p);
        }
    }

    // --- Move resolution (moves.py mirror) ---

    pub(crate) fn priority_sort(
        &self,
        candidates: &[usize],
        m_index: &[i8],
        m_source: &[i16],
        m_dest: &[i16],
    ) -> Vec<usize> {
        let mut sorted: Vec<usize> = candidates.to_vec();
        sorted.sort_by_key(|&i| {
            let p = m_index[i];
            let dest = m_dest[i] as usize;
            let source = m_source[i] as usize;
            (
                if self.ownership[dest] == p { 0 } else { 1 },
                if !self.is_general_attack(dest, p) { 0 } else { 1 },
                -self.armies[source],
                p as i32,
            )
        });
        sorted
    }

    pub(crate) fn dependency_loop(
        &self,
        sorted_candidates: &[usize],
        m_source: &[i16],
        m_dest: &[i16],
    ) -> Vec<usize> {
        let mut remaining: Vec<usize> = sorted_candidates.to_vec();
        let mut result: Vec<usize> = Vec::with_capacity(remaining.len());
        while !remaining.is_empty() {
            let mut taken = false;
            for j in 0..remaining.len() {
                let m = remaining[j];
                let src = m_source[m];
                let blocked = remaining
                    .iter()
                    .enumerate()
                    .any(|(k, &other)| k != j && m_dest[other] == src);
                if !blocked {
                    result.push(remaining.remove(j));
                    taken = true;
                    break;
                }
            }
            if !taken {
                result.push(remaining.remove(0));
            }
        }
        result
    }

    pub(crate) fn is_general_attack(&self, dest: usize, mover: i8) -> bool {
        for (p, &g) in self.generals.iter().enumerate() {
            if g == dest as i32 {
                return p as i8 != mover;
            }
        }
        false
    }

    // --- AFK processing + step orchestrator + snapshot ---

    pub(crate) fn process_pending_afks(&mut self, a_timestep: &[i32], a_index: &[i8]) {
        while self.afks_cursor < a_timestep.len()
            && a_timestep[self.afks_cursor] <= self.timestep
        {
            let p = a_index[self.afks_cursor] as usize;
            if self.alive[p] {
                self.kill_player(p);
            } else {
                self.try_neutralize_player(p);
            }
            self.afks_cursor += 1;
            if self.alive_count <= 1 {
                break;
            }
        }
    }

    pub(crate) fn snapshot(&mut self) -> Result<(), ArmyOverflow> {
        let max = self.armies.iter().copied().max().unwrap_or(0);
        if max > ARMY_MAX {
            return Err(ArmyOverflow {
                timestep: self.timestep,
                value: max,
            });
        }
        self.snapshots_ownership.push(self.ownership.clone());
        self.snapshots_armies
            .push(self.armies.iter().map(|&a| a as i16).collect());
        self.snapshots_cities_mask.push(self.cities_mask.clone());
        Ok(())
    }

    pub(crate) fn step(
        &mut self,
        m_timestep: &[i32],
        m_index: &[i8],
        m_source: &[i16],
        m_dest: &[i16],
        m_is50: &[u8],
        a_timestep: &[i32],
        a_index: &[i8],
    ) -> Result<bool, ArmyOverflow> {
        if self.alive_count <= 1 {
            return Ok(false);
        }

        self.process_pending_afks(a_timestep, a_index);
        self.buffer_pending_moves(m_timestep, m_index);
        let candidates = self.select_candidates(m_index, m_source, m_dest);

        let any_ran = {
            let sorted = self.priority_sort(&candidates, m_index, m_source, m_dest);
            let ordered = self.dependency_loop(&sorted, m_source, m_dest);
            let mut ran = false;
            for &i in &ordered {
                if self.is_valid(i, m_index, m_source, m_dest) {
                    self.execute_attack(i, m_index, m_source, m_dest, m_is50);
                    ran = true;
                }
            }
            ran
        };

        self.updates_since_move = if any_ran {
            0
        } else {
            self.updates_since_move + 1
        };
        if self.updates_since_move > MAX_ALL_AFK_TIMESTEPS || self.timestep > MAX_GAME_TIMESTEPS {
            self.kill_all_but_strongest_impl();
        }

        self.timestep += 1;
        self.apply_production_impl();
        self.snapshot()?;
        Ok(true)
    }

    pub(crate) fn build_initial(
        map_size: usize,
        num_players: usize,
        mountains: &[i32],
        initial_cities: &[i32],
        initial_city_armies: &[i32],
        initial_generals: &[i32],
        initial_neutrals: &[i32],
        initial_neutral_armies: &[i32],
    ) -> Self {
        let mut ownership = vec![-1i8; map_size];
        let mut armies = vec![0i32; map_size];
        let mut cities_mask = vec![0u8; map_size];

        for &m in mountains {
            ownership[m as usize] = -2;
        }

        let mut cities: Vec<i32> = Vec::with_capacity(initial_cities.len());
        for (&idx, &army) in initial_cities.iter().zip(initial_city_armies.iter()) {
            cities.push(idx);
            cities_mask[idx as usize] = 1;
            armies[idx as usize] = army;
        }

        let mut generals: Vec<i32> = Vec::with_capacity(num_players);
        let mut alive = vec![true; num_players];
        let mut alive_count = num_players;
        for (p, &gen) in initial_generals.iter().enumerate() {
            if gen >= 0 {
                ownership[gen as usize] = p as i8;
                armies[gen as usize] = 1;
                generals.push(gen);
            } else {
                generals.push(-1);
                alive[p] = false;
                alive_count -= 1;
            }
        }

        for (&idx, &army) in initial_neutrals.iter().zip(initial_neutral_armies.iter()) {
            armies[idx as usize] = army;
        }

        State {
            ownership,
            armies,
            cities_mask,
            cities,
            generals,
            alive,
            has_kill: vec![false; num_players],
            input_buffer: (0..num_players).map(|_| VecDeque::new()).collect(),
            timestep: 0,
            num_players,
            alive_count,
            updates_since_move: 0,
            afks_cursor: 0,
            moves_cursor: 0,
            damage_off_all: vec![0i32; num_players * num_players],
            death_events: Vec::new(),
            capture_events: Vec::new(),
            neutralize_events: Vec::new(),
            snapshots_ownership: Vec::new(),
            snapshots_armies: Vec::new(),
            snapshots_cities_mask: Vec::new(),
            map_size,
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn damage_to_pyarray2<'py>(
    py: Python<'py>,
    flat: &[i32],
    p: usize,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    use numpy::ndarray::Array2;
    let arr = Array2::from_shape_vec((p, p), flat.to_vec())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("damage reshape: {e}")))?;
    Ok(numpy::PyArray2::from_owned_array(py, arr))
}

pub fn army_overflow_pyerr(py: Python<'_>, e: ArmyOverflow) -> PyErr {
    let msg = format!("army > {} at t={}", ARMY_MAX, e.timestep);
    match py
        .import("replay_parser.errors")
        .and_then(|m| m.getattr("ArmyOverflowError"))
        .and_then(|cls| cls.call1((msg.clone(),)))
    {
        Ok(exc) => PyErr::from_value(exc),
        Err(_) => pyo3::exceptions::PyRuntimeError::new_err(msg),
    }
}
