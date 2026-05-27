"""Tunable constants for EvalBot behavior."""

from __future__ import annotations

from dataclasses import dataclass


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
    KILLSHOT_MARGIN_BASE: int = 10
    KILLSHOT_MARGIN_PER_STEP: float = 2
    RACE_MARGIN: int = 4

    # -- attack --
    ATTACK_MAX_PLAN_TICKS: int = 20
    ATTACK_SWITCH_THRESHOLD: float = 0.5
    # Renamed from `RISK_FRACTION` in design docs / specs
    ATTACK_COMMIT_FRACTION: float = 0.4

    # -- attack: target scoring --
    TARGET_SCORE_W_WEAKNESS: float = 1.0
    TARGET_SCORE_W_PROXIMITY: float = 0.8
    PROXIMITY_HALF_DIST: float = 12
    PROXIMITY_K: int = 10
    TARGET_SCORE_W_GROWING: float = 0
    TARGET_SCORE_W_CONFLICT: float = 0

    # -- conflict detection --
    CONFLICT_WINDOW: int = 20
    CONFLICT_THRESHOLD: int = 3
    CONFLICT_MATCH_TOL: int = 5

    # -- expand / explore --
    EXPAND_EXPLORE_THRESHOLD: float = 0.4
    EXPLORE_FOG_DEPTH: int = 3
    EXPLORE_MAX_PLAN_TICKS: int = 15
    EXPLORE_MIN_ARMY: int = 5
