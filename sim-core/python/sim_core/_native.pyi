from typing import Any

import numpy as np

def ping() -> None: ...

class DeathEvent:
    timestep: int
    player: int

class CaptureEvent:
    timestep: int
    captor: int
    captured: int

class NeutralizeEvent:
    timestep: int
    player: int

class State:
    @classmethod
    def from_python(cls, py_state: Any) -> "State": ...

    # Read-only views (each call returns a fresh numpy array)
    @property
    def timestep(self) -> int: ...
    @property
    def alive_count(self) -> int: ...
    @property
    def moves_cursor(self) -> int: ...
    @property
    def updates_since_move(self) -> int: ...
    @property
    def ownership(self) -> np.ndarray: ...
    @property
    def armies(self) -> np.ndarray: ...
    @property
    def cities_mask(self) -> np.ndarray: ...
    @property
    def actions_source(self) -> np.ndarray: ...
    @property
    def actions_dest(self) -> np.ndarray: ...
    @property
    def actions_is50(self) -> np.ndarray: ...
    @property
    def input_buffer_lengths(self) -> list[int]: ...
    @property
    def input_buffer_contents(self) -> list[list[int]]: ...

    # Step body methods
    def apply_production(self) -> None: ...
    def buffer_pending_moves(self, m_timestep: np.ndarray, m_index: np.ndarray) -> None: ...
    def select_candidates(
        self, m_index: np.ndarray, m_source: np.ndarray, m_dest: np.ndarray
    ) -> list[int]: ...
    def is_valid(
        self,
        move_idx: int,
        m_index: np.ndarray,
        m_source: np.ndarray,
        m_dest: np.ndarray,
    ) -> bool: ...
    def record_action(
        self,
        move_idx: int,
        m_index: np.ndarray,
        m_source: np.ndarray,
        m_dest: np.ndarray,
        m_is50: np.ndarray,
    ) -> None: ...
