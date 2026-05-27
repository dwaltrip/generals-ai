from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import sim_core


class Policy(Protocol):
    def init_for_game(self, state: sim_core.State, static: Any) -> None: ...
    def act(self, state: sim_core.State, static: Any) -> tuple[int, int, int]: ...


@dataclass
class PlayerStats:
    land: int = 0
    army: int = 0
    n_moved: int = 0
    n_passed: int = 0


@dataclass
class GameResult:
    winner: int | None
    game_length: int
    player_stats: list[PlayerStats] = field(default_factory=list)
    state: sim_core.State | None = None
