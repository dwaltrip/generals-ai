"""Defend gate — highest-priority interrupt.

Detects incursion threats and responds with gather_path to intercept
or rally to the general.
"""

from __future__ import annotations

from eval_bot.bfs import bfs_path
from eval_bot.gather import gather_path
from eval_bot.plan import DefendPlan
from eval_bot.world_model import PlayerView

INTERCEPT_LOOKAHEAD = 3


def _pick_most_severe(view: PlayerView) -> int | None:
    """Pick the most threatening player: closest to general, then largest army."""
    threats = [p for p, active in view.incursion_threats.items() if active]
    if not threats:
        return None
    return min(threats, key=lambda p: (
        view.threat_windows[p][-1].dist_to_general,
        -view.threat_windows[p][-1].army,
    ))


def try_defend(view: PlayerView) -> DefendPlan | None:
    most_severe = _pick_most_severe(view)
    if most_severe is None:
        return None

    tw = view.threat_windows[most_severe][-1]

    # severity → target
    gen_r, gen_c = divmod(view.gen_flat, view.W)
    my_gen_army = int(view.arm[gen_r, gen_c])
    if tw.army > my_gen_army:
        target = view.gen_flat
    else:
        path = bfs_path(tw.pos, view.gen_flat, view.passable, view.H, view.W)
        if not path:
            return None
        idx = min(INTERCEPT_LOOKAHEAD, len(path) - 1)
        target = path[idx]

    # time budget
    max_moves = tw.dist_to_general - INTERCEPT_LOOKAHEAD
    if max_moves <= 0:
        return None

    result = gather_path(
        view, target, max_moves=max_moves, min_army=None, gate="defend",
    )
    if result is None:
        return None

    return DefendPlan(moves=result.moves, target_player=most_severe)


def should_clear_defend(view: PlayerView, plan: DefendPlan) -> bool:
    if not any(view.incursion_threats.values()):
        return True
    new_most_severe = _pick_most_severe(view)
    if new_most_severe != plan.target_player:
        return True
    return False
