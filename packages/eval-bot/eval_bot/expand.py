"""ExpandOrExplore gate — the bot's default mode.

Two sub-modes switched by the ExpandableTiles / UnoccupiedFrontier ratio:
  - Expand: thin BFS crawl into adjacent empty tiles (cheap, no gather).
  - Explore: gather_path push toward the least-explored sector.

Expand returns SingleMove. Explore returns ExplorePlan.
"""

from __future__ import annotations

import numpy as np

from eval_bot.bfs import bfs_distances, bfs_distances_multi
from eval_bot.bot_config import BotConfig
from eval_bot.gather import gather_path
from eval_bot.plan import ExplorePlan, GateResult, SingleMove
from eval_bot.world_model import PlayerView

NEUTRAL = -1

MOVE_EXPAND = "expand"
MOVE_EXPLORE = "explore"


def _expandable_tiles(
    own: np.ndarray, arm: np.ndarray, my_slot: int, H: int, W: int,
) -> np.ndarray:
    """Bool [H, W] mask of owned tiles with army >= 2 adjacent to at least
    one neutral 0-garrison tile."""
    mine = own == my_slot
    has_army = arm >= 2
    empty = (own == NEUTRAL) & (arm == 0)

    adj_empty = np.zeros((H, W), dtype=bool)
    adj_empty[1:, :] |= empty[:-1, :]
    adj_empty[:-1, :] |= empty[1:, :]
    adj_empty[:, 1:] |= empty[:, :-1]
    adj_empty[:, :-1] |= empty[:, 1:]

    return mine & has_army & adj_empty


def _unoccupied_frontier(
    own: np.ndarray, my_slot: int, H: int, W: int,
) -> np.ndarray:
    """Bool [H, W] mask of owned tiles adjacent to at least one neutral tile."""
    mine = own == my_slot
    neutral = own == NEUTRAL

    adj_neutral = np.zeros((H, W), dtype=bool)
    adj_neutral[1:, :] |= neutral[:-1, :]
    adj_neutral[:-1, :] |= neutral[1:, :]
    adj_neutral[:, 1:] |= neutral[:, :-1]
    adj_neutral[:, :-1] |= neutral[:, 1:]

    return mine & adj_neutral


def _pick_expand_move(
    expandable: np.ndarray,
    own: np.ndarray,
    arm: np.ndarray,
    gen_dist: np.ndarray,
    H: int,
    W: int,
) -> tuple[int, int, int] | None:
    """Pick the best thin expand move: farthest expandable tile from the
    general (pushes outward), move onto its adjacent empty neighbor."""
    exp_flat = expandable.reshape(-1)
    candidates = np.where(exp_flat)[0]
    if len(candidates) == 0:
        return None

    # sort by distance from general, descending (farthest first)
    dists = gen_dist[candidates]
    # gen_dist is -1 for unreachable — put those last
    dists_sortable = np.where(dists >= 0, dists, -1)
    order = np.argsort(-dists_sortable)
    candidates = candidates[order]

    empty_flat = ((own == NEUTRAL) & (arm == 0)).reshape(-1)
    for src in candidates:
        r, c = divmod(int(src), W)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < H and 0 <= nc < W:
                dst = nr * W + nc
                if empty_flat[dst]:
                    return (int(src), dst, 0)

    return None


def _sector_of(dr: int, dc: int) -> int:
    """Assign a (dr, dc) offset to one of 4 quadrants: 0=NW, 1=NE, 2=SW, 3=SE."""
    if dr <= 0 and dc <= 0:
        return 0  # NW
    if dr <= 0 and dc > 0:
        return 1  # NE
    if dr > 0 and dc <= 0:
        return 2  # SW
    return 3  # SE


def _pick_explore_plan(view: PlayerView, cfg: BotConfig) -> ExplorePlan | None:
    """Gather toward the least-explored sector using gather_path."""
    own = view.own
    my_slot = view.my_slot
    passable = view.passable
    gen_flat = view.gen_flat
    H, W = view.H, view.W

    gen_r, gen_c = divmod(gen_flat, W)

    # score each sector by count of passable non-owned tiles
    passable_2d = passable.reshape(H, W)
    not_mine = own != my_slot
    sector_scores = [0, 0, 0, 0]

    for r in range(H):
        for c in range(W):
            if passable_2d[r, c] and not_mine[r, c]:
                sector_scores[_sector_of(r - gen_r, c - gen_c)] += 1

    if max(sector_scores) == 0:
        return None

    best_sector = int(np.argmax(sector_scores))

    # frontier tiles in the winning sector
    frontier = _unoccupied_frontier(own, my_slot, H, W)
    frontier_in_sector: list[int] = []
    for r in range(H):
        for c in range(W):
            if frontier[r, c] and _sector_of(r - gen_r, c - gen_c) == best_sector:
                frontier_in_sector.append(r * W + c)

    if not frontier_in_sector:
        return None

    # multi-source BFS from frontier to find target at depth
    dist_from_frontier = bfs_distances_multi(frontier_in_sector, passable, H, W)

    own_flat = own.reshape(-1)
    candidates_by_depth: dict[int, list[int]] = {}
    for cell in range(H * W):
        d = int(dist_from_frontier[cell])
        if d <= 0 or d > cfg.EXPLORE_FOG_DEPTH:
            continue
        if own_flat[cell] == my_slot:
            continue
        candidates_by_depth.setdefault(d, []).append(cell)

    if not candidates_by_depth:
        return None

    best_depth = max(candidates_by_depth)
    targets = candidates_by_depth[best_depth]
    target = int(np.random.choice(targets))

    result = gather_path(
        view, cfg, target,
        max_moves=cfg.EXPLORE_MAX_PLAN_TICKS,
        min_army=cfg.EXPLORE_MIN_ARMY,
        gate="explore",
    )
    if result is None:
        return None

    return ExplorePlan(moves=result.moves)


def _to_single(move: tuple[int, int, int], gate: str) -> SingleMove:
    return SingleMove(src=move[0], dst=move[1], is50=move[2], gate=gate)


def pick_expand_or_explore(view: PlayerView, cfg: BotConfig) -> GateResult:
    """Top-level ExpandOrExplore gate. Returns SingleMove, ExplorePlan, or None."""
    own = view.own
    arm = view.arm
    my_slot = view.my_slot
    passable = view.passable
    gen_flat = view.gen_flat
    H = view.H
    W = view.W

    expandable = _expandable_tiles(own, arm, my_slot, H, W)
    frontier = _unoccupied_frontier(own, my_slot, H, W)

    expandable_count = int(expandable.sum())
    frontier_count = int(frontier.sum())

    if frontier_count == 0:
        return None

    if expandable_count > 0 and expandable_count / frontier_count >= cfg.EXPAND_EXPLORE_THRESHOLD:
        gen_dist = bfs_distances(gen_flat, passable, H, W)
        move = _pick_expand_move(expandable, own, arm, gen_dist, H, W)
        if move is not None:
            return _to_single(move, MOVE_EXPAND)

    # explore mode
    plan = _pick_explore_plan(view, cfg)
    if plan is not None:
        return plan

    # explore couldn't build a plan — fall back to expand if any expandable tiles
    if expandable_count > 0:
        gen_dist = bfs_distances(gen_flat, passable, H, W)
        move = _pick_expand_move(expandable, own, arm, gen_dist, H, W)
        if move is not None:
            return _to_single(move, MOVE_EXPAND)

    return None
