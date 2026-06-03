"""Self-play convenience wrapper: two NN agents (same weights) on a 2-player map."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bc.inference import BCModelHandle, BCPerspective

from game_runner.runner import run_game
from self_play.nn_agent import NNAgent


if TYPE_CHECKING:
    from game_types import StaticMap

    import sim_core


def play_game(
    handle: BCModelHandle,
    map_data: StaticMap,
    max_turns: int = 2000,
    force_move: bool = False,
    sample: bool = False,
    temperature: float = 1.0,
    progress_interval: int = 50,
) -> tuple[sim_core.State, list[NNAgent]]:
    """Play one 2-player self-play game (both slots driven by `handle`).

    Returns (State, agents) for the CLI and tests.
    """
    agents = [
        NNAgent(
            handle,
            BCPerspective(
                p,
                handle.device,
                handle.model.cfg.obs,
                force_move=force_move,
                sample=sample,
                temperature=temperature,
            ),
        )
        for p in (0, 1)
    ]
    result = run_game(
        agents, map_data,
        max_turns=max_turns,
        progress_interval=progress_interval,
    )
    assert result.state is not None
    return result.state, agents
