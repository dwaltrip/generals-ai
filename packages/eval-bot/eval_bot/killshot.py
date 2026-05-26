"""Kill-shot gate — capture a revealed enemy general.

Highest-value, most time-sensitive target. Fully sticky while in-flight.
"""

from __future__ import annotations

from eval_bot.bot_config import BotConfig
from eval_bot.plan import KillshotPlan
from eval_bot.world_model import PlayerView


def try_killshot(view: PlayerView, cfg: BotConfig) -> KillshotPlan | None:
    # TODO: iterate revealed generals, build candidate via gather_path,
    # check margin feasibility
    return None
