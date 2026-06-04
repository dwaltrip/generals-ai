"""Minimal game loop: N EvalBots play to termination on a seed map.

Modeled on self_play.driver.play_game but supports any player count
and uses EvalBot instead of ModelAgent.
"""

from __future__ import annotations

from eval_bot.bot import EvalBot
from eval_bot.bot_config import BotConfig
from game_types import StaticMap
import sim_core


def play_game(
    map_data: StaticMap,
    cfg: BotConfig | None = None,
    max_ticks: int = 2000,
    progress_interval: int = 50,
) -> tuple[sim_core.State, list[EvalBot]]:
    """Play one game with EvalBots on every slot. Returns the final state
    and the list of bots (for diagnostics).
    """
    state = sim_core.new_state(map_data)
    num_players = state.num_players

    bot_cfg = cfg or BotConfig()
    bots = [
        EvalBot(perspective_slot=p, cfg=bot_cfg)
        for p in range(num_players)
    ]
    for bot in bots:
        bot.init_for_game(state, map_data)

    while state.alive_count > 1 and state.timestep < max_ticks:
        moves: list[tuple[int, int, int, int]] = []
        for p, bot in enumerate(bots):
            if not state.alive[p]:
                continue
            view = bot.world.update(state, map_data)
            src, dst, is50 = bot.act(view)
            if src == -1:
                continue
            moves.append((p, src, dst, is50))
        state.step_tick(moves=moves, afks=[])

        if progress_interval > 0 and state.timestep % progress_interval == 0:
            _print_progress(state, bots)

    return state, bots


def _print_progress(state: sim_core.State, bots: list[EvalBot]) -> None:
    own = state.ownership
    arm = state.armies
    alive_chr = "".join("T" if a else "F" for a in state.alive)
    parts = []
    for p, bot in enumerate(bots):
        mask = own == p
        land = int(mask.sum())
        army = int((arm * mask).sum())
        parts.append(f"p{p}: land={land:3d} army={army:4d} move={bot.n_moved:4d}")
    print(f"[t={state.timestep:4d}] alive=[{alive_chr}] {' | '.join(parts)}")
