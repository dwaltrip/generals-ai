"""Multi-tick plan — a pre-computed move sequence emitted one move per tick."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Plan:
    gate: str
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
