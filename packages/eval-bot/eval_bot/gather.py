"""gather_path — assemble a stack and sweep it to a destination.

Stub for now. Implementation in section 5.
"""

from __future__ import annotations

from eval_bot.plan import GatherResult
from eval_bot.world_model import PlayerView


def gather_path(
    view: PlayerView,
    target: int,
    max_moves: int | None = None,
    min_army: int | None = None,
    gate: str = "",
) -> GatherResult | None:
    # TODO (section 5): source selection, BFS path, side-gathers, sweep,
    # reserve/split, army estimate, feasibility check.
    return None
