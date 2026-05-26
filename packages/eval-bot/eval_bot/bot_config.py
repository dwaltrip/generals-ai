"""Tunable constants for EvalBot behavior."""

from __future__ import annotations

from dataclasses import dataclass

BOT_CFG_PLEASE_CHANGE_ME = 999


@dataclass
class BotConfig:
    # -- threat detection --
    THREAT_WINDOW_LEN: int = 8
    CLOSING_THRESHOLD: float = -0.3
    EATING_RADIUS: int = 3
    EATING_MIN: int = 3

    # -- gather_path --
    RESERVE_FRACTION: float = 0.25
    SOURCE_DIST_WEIGHT: float = 1.0
    SIDE_GATHER_FRACTION: float = 0.5

    # -- defend --
    INTERCEPT_LOOKAHEAD: int = 3

    # -- kill-shot --
    KILLSHOT_MARGIN_BASE: int = BOT_CFG_PLEASE_CHANGE_ME        # guesstimate: 3
    KILLSHOT_MARGIN_PER_STEP: float = BOT_CFG_PLEASE_CHANGE_ME  # guesstimate: 0.5
    RACE_MARGIN: int = BOT_CFG_PLEASE_CHANGE_ME                 # guesstimate: 2

    # -- attack --
    ATTACK_MAX_PLAN_TICKS: int = BOT_CFG_PLEASE_CHANGE_ME       # guesstimate: 20
    ATTACK_SWITCH_THRESHOLD: float = BOT_CFG_PLEASE_CHANGE_ME   # guesstimate: 0.3
    RISK_FRACTION: float = BOT_CFG_PLEASE_CHANGE_ME             # guesstimate: 0.5

    # -- attack: target scoring --
    TARGET_SCORE_W_WEAKNESS: float = BOT_CFG_PLEASE_CHANGE_ME   # guesstimate: 1.0
    TARGET_SCORE_W_PROXIMITY: float = BOT_CFG_PLEASE_CHANGE_ME  # guesstimate: 0.5
    TARGET_SCORE_W_GROWING: float = BOT_CFG_PLEASE_CHANGE_ME    # guesstimate: 0.3
    TARGET_SCORE_W_CONFLICT: float = BOT_CFG_PLEASE_CHANGE_ME   # guesstimate: 0.5

    # -- conflict detection --
    CONFLICT_WINDOW: int = 20
    CONFLICT_THRESHOLD: int = 3
    CONFLICT_MATCH_TOL: int = 5

    # -- expand / explore --
    EXPAND_EXPLORE_THRESHOLD: float = 0.4
    EXPLORE_FOG_DEPTH: int = 3
    EXPLORE_MAX_PLAN_TICKS: int = 15
    EXPLORE_MIN_ARMY: int = 5
