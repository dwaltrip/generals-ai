"""Kill-shot gate — capture a revealed enemy general.

Highest-value, most time-sensitive target. Fully sticky while in-flight.
"""

from __future__ import annotations

import math

from eval_bot.bot_config import BotConfig
from eval_bot.gather import gather_path
from eval_bot.plan import KillshotPlan
from eval_bot.world_model import PlayerView


def try_killshot(view: PlayerView, cfg: BotConfig) -> KillshotPlan | None:
    mem = view.mem

    for p in range(len(mem.general_locations)):
        if p == view.my_slot:
            continue
        gen_pos = int(mem.general_locations[p])
        if gen_pos < 0:
            continue
        if not view.alive[p]:
            continue

        # same-player killshot bypasses reserve (§6b.2)
        is_same_player_threat = view.incursion_threats.get(p, False)

        candidate = gather_path(
            view, cfg, gen_pos,
            max_moves=None,
            min_army=None,
            bypass_reserve=is_same_player_threat,
        )
        if candidate is None:
            continue

        travel_time = len(candidate.moves)
        margin = cfg.KILLSHOT_MARGIN_BASE + math.ceil(
            cfg.KILLSHOT_MARGIN_PER_STEP * travel_time,
        )

        gen_r, gen_c = divmod(gen_pos, view.W)
        gen_army = int(mem.last_seen_armies[gen_r, gen_c])

        if candidate.arriving_army >= gen_army + margin:
            return KillshotPlan(
                moves=candidate.moves,
                target_player=p,
                target_general=gen_pos,
            )

    return None
