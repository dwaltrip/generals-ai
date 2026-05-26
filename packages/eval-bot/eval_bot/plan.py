"""Gate result types — plans and single moves."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SingleMove:
    src: int
    dst: int
    is50: int
    gate: str


@dataclass
class Plan:
    moves: list[tuple[int, int, int]]
    cursor: int = field(default=0, init=False)

    def next_move(self) -> tuple[int, int, int] | None:
        if self.cursor >= len(self.moves):
            return None
        move = self.moves[self.cursor]
        self.cursor += 1
        return move

    @property
    def is_complete(self) -> bool:
        return self.cursor >= len(self.moves)


@dataclass
class DefendPlan(Plan):
    target_player: int = -1


@dataclass
class ExplorePlan(Plan):
    pass


@dataclass
class GatherResult:
    """Raw output of gather_path — caller wraps in the appropriate Plan variant."""
    moves: list[tuple[int, int, int]]
    arriving_army: int


GateResult = Plan | SingleMove | None
