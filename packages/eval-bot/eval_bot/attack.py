"""Attack gate — project force into enemy territory.

Target selection via TargetScore. Somewhat-sticky via hysteresis
(current target gets a scoring bonus). Also contains the scoring
helpers: RelativeStrength, ConflictFlag, TargetScore.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from eval_bot.bfs import bfs_distances
from eval_bot.bot_config import BotConfig
from eval_bot.gather import gather_path, reserve_amount
from eval_bot.plan import AttackPlan
from eval_bot.world_model import PlayerView


class RelativeStrength(NamedTuple):
    army: float
    land: float


def relative_strength(view: PlayerView, p: int) -> RelativeStrength:
    alive_others = [
        q for q in range(len(view.alive))
        if view.alive[q] and q != p
    ]
    if not alive_others:
        return RelativeStrength(army=1.0, land=1.0)

    mean_army = sum(int(view.army[q]) for q in alive_others) / len(alive_others)
    mean_land = sum(int(view.land[q]) for q in alive_others) / len(alive_others)
    return RelativeStrength(
        army=float(view.army[p]) / max(1, mean_army),
        land=float(view.land[p]) / max(1, mean_land),
    )


def conflict_flag(view: PlayerView, cfg: BotConfig, p: int) -> bool:
    """Is player p currently fighting another player (mirrored land deltas)?"""
    t = view.timestep
    if t < cfg.CONFLICT_WINDOW:
        return False

    land_now = view.mem.land_count_history[t]
    land_prev = view.mem.land_count_history[t - cfg.CONFLICT_WINDOW]
    delta_p = int(land_now[p]) - int(land_prev[p])

    if delta_p >= -cfg.CONFLICT_THRESHOLD:
        return False

    for q in range(len(view.alive)):
        if q == p or not view.alive[q]:
            continue
        delta_q = int(land_now[q]) - int(land_prev[q])
        if delta_q <= cfg.CONFLICT_THRESHOLD:
            continue
        if abs(delta_p + delta_q) < cfg.CONFLICT_MATCH_TOL:
            return True

    return False


def _target_score(
    view: PlayerView,
    cfg: BotConfig,
    p: int,
    rs: RelativeStrength,
    dist_from_gen: np.ndarray,
) -> float:
    """Score how attractive player p is as an attack target."""
    # hard gate: don't initiate against much stronger players
    # here, set a bit arbitrarily to 50%
    # TODO: this should be a much more sophisticed calculation.
    if rs.army > 1.5:
        return 0.0

    # weakness: inverse of army ratio (weaker = higher score)
    weakness = 1.0 / max(0.1, rs.army)

    # proximity: exponential decay of average BFS distance of K nearest visible tiles
    fog_own = view.own.reshape(-1)
    p_tiles = np.where(fog_own == p)[0]
    if len(p_tiles) == 0:
        return 0.0
    dists = dist_from_gen[p_tiles]
    reachable = dists[dists >= 0]
    if len(reachable) == 0:
        return 0.0
    nearest_k = np.sort(reachable)[:cfg.PROXIMITY_K]
    avg_dist = float(nearest_k.mean())
    proximity = math.exp(-avg_dist / cfg.PROXIMITY_HALF_DIST)

    # growing: land delta > 0 means closing window of opportunity
    t = view.timestep
    if t >= cfg.CONFLICT_WINDOW:
        land_now = int(view.mem.land_count_history[t][p])
        land_prev = int(view.mem.land_count_history[t - cfg.CONFLICT_WINDOW][p])
        growing = float(land_now - land_prev > 0)
    else:
        growing = 0.0

    # conflict: player is distracted, don't provoke
    in_conflict = float(conflict_flag(view, cfg, p))

    return (
        cfg.TARGET_SCORE_W_WEAKNESS * weakness
        + cfg.TARGET_SCORE_W_PROXIMITY * proximity
        + cfg.TARGET_SCORE_W_GROWING * growing
        - cfg.TARGET_SCORE_W_CONFLICT * in_conflict
    )


def _pick_primary_target(
    view: PlayerView,
    cfg: BotConfig,
    dist_from_gen: np.ndarray,
) -> tuple[int, float] | None:
    """Pick the best attack target. Returns (player_slot, score) or None."""
    best_p = -1
    best_score = 0.0

    for p in view.contact_set:
        rs = relative_strength(view, p)
        score = _target_score(view, cfg, p, rs, dist_from_gen)
        if score > best_score:
            best_score = score
            best_p = p

    if best_p == -1:
        return None
    return best_p, best_score


def _pick_attack_target_tile(
    view: PlayerView, target_player: int, dist_from_gen: np.ndarray,
) -> int | None:
    """Find the tile to gather toward for attacking target_player."""
    # if we know their general, go for it
    gen_pos = int(view.mem.general_locations[target_player])
    if gen_pos >= 0:
        return gen_pos

    # otherwise, their nearest visible tile to our general
    fog_own = view.own.reshape(-1)
    p_tiles = np.where(fog_own == target_player)[0]
    if len(p_tiles) == 0:
        return None
    dists = dist_from_gen[p_tiles]
    reachable_mask = dists >= 0
    if not reachable_mask.any():
        return None
    best_idx = int(np.where(reachable_mask, dists, np.iinfo(np.int32).max).argmin())
    return int(p_tiles[best_idx])


def try_attack(view: PlayerView, cfg: BotConfig) -> AttackPlan | None:
    if not view.contact_set:
        return None

    dist_from_gen = bfs_distances(view.gen_flat, view.passable, view.H, view.W)

    result = _pick_primary_target(view, cfg, dist_from_gen)
    if result is None:
        return None
    target_player, _score = result

    target_tile = _pick_attack_target_tile(view, target_player, dist_from_gen)
    if target_tile is None:
        return None

    # commit budget: fraction of mobile army (spec §5.1, §6c.4)
    total_army = int(view.army[view.my_slot])
    mobile = total_army - reserve_amount(view, cfg)
    commit = int(cfg.ATTACK_COMMIT_FRACTION * mobile)

    candidate = gather_path(
        view, cfg, target_tile,
        max_moves=cfg.ATTACK_MAX_PLAN_TICKS,
        min_army=commit,
    )
    if candidate is None:
        return None

    return AttackPlan(moves=candidate.moves, target_player=target_player)


def should_clear_attack(
    view: PlayerView, cfg: BotConfig, plan: AttackPlan,
) -> bool:
    """Clear if a new target outscores the current one by the switch threshold."""
    if not view.contact_set:
        return True

    dist_from_gen = bfs_distances(view.gen_flat, view.passable, view.H, view.W)

    result = _pick_primary_target(view, cfg, dist_from_gen)
    if result is None:
        return True
    new_best, new_score = result

    if new_best == plan.target_player:
        return False

    # current target's score
    rs = relative_strength(view, plan.target_player)
    current_score = _target_score(view, cfg, plan.target_player, rs, dist_from_gen)

    return new_score > current_score + cfg.ATTACK_SWITCH_THRESHOLD
