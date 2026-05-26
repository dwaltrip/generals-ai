"""Attack gate — project force into enemy territory.

Target selection via ThreatScore/TargetScore. Somewhat-sticky via
hysteresis (current target gets a scoring bonus).
"""

from __future__ import annotations

from eval_bot.bot_config import BotConfig
from eval_bot.plan import AttackPlan
from eval_bot.world_model import PlayerView


def try_attack(view: PlayerView, cfg: BotConfig) -> AttackPlan | None:
    # TODO: score contacts via TargetScore, pick PrimaryTarget,
    # gather_path to nearest frontier tile or revealed general
    return None


def should_clear_attack(view: PlayerView, cfg: BotConfig, plan: AttackPlan) -> bool:
    # TODO: recompute TargetScore, clear if new best outscores current
    # by ATTACK_SWITCH_THRESHOLD
    return False
