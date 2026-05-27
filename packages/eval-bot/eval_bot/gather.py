"""gather_path — assemble a stack and sweep it to a destination.

Shared primitive used by Defend, Explore, Attack, and Kill-shot.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from eval_bot.bfs import bfs_distances, bfs_path
from eval_bot.bot_config import BotConfig
from eval_bot.plan import GatherResult
from eval_bot.world_model import PlayerView


# TODO: very crude model of army budgeting — a single global fraction
# doesn't account for board position, threat proximity, or how much
# army is actually reachable from the gather source. Revisit when tuning.
def reserve_amount(view: PlayerView, cfg: BotConfig) -> int:
    total_army = int(view.army[view.my_slot])
    return int(cfg.RESERVE_FRACTION * total_army)


def mobile_army(
    tiles: list[int],
    arm_flat: np.ndarray,
    own_flat: np.ndarray,
    my_slot: int,
) -> int:
    """Sum of (army - 1) for each owned tile — the army that can move off."""
    total = 0
    for t in tiles:
        if own_flat[t] != my_slot:
            warnings.warn(f"mobile_army: tile {t} not owned by slot {my_slot}")
            continue
        a = int(arm_flat[t])
        if a >= 2:
            total += a - 1
    return total


def _effective_army_general(
    gen_army: int, reserve_amount: int, bypass_reserve: bool,
) -> tuple[int, bool]:
    """Returns (effective_army, use_split) for the general tile."""
    if bypass_reserve:
        return gen_army - 1, False
    if math.ceil(gen_army / 2) >= reserve_amount:
        return gen_army // 2, True
    return 0, False


def _select_source(
    view: PlayerView,
    cfg: BotConfig,
    target: int,
    max_moves: int | None,
    reserve_amount: int,
    bypass_reserve: bool,
) -> tuple[int, bool] | None:
    """Pick the best source tile. Returns (source_flat, use_split) or None."""
    own_flat = view.own.reshape(-1)
    arm_flat = view.arm.reshape(-1)

    dist_from_target = bfs_distances(target, view.passable, view.H, view.W)

    best_score = -1
    best_source = -1
    best_split = False

    for tile in range(len(own_flat)):
        if own_flat[tile] != view.my_slot:
            continue
        if arm_flat[tile] < 2:
            continue
        d = dist_from_target[tile]
        if d < 0:
            continue
        if max_moves is not None and d > max_moves:
            continue

        if tile == view.gen_flat:
            eff, use_split = _effective_army_general(
                int(arm_flat[tile]), reserve_amount, bypass_reserve,
            )
        else:
            eff = int(arm_flat[tile]) - 1
            use_split = False

        if eff <= 0:
            continue

        score = eff - cfg.SOURCE_DIST_WEIGHT * d
        if score > best_score:
            best_score = score
            best_source = tile
            best_split = use_split

    if best_source == -1:
        return None
    return best_source, best_split


def _find_side_gather_candidates(
    path: list[int],
    own_flat: np.ndarray,
    arm_flat: np.ndarray,
    my_slot: int,
    H: int,
    W: int,
) -> list[tuple[int, int, int]]:
    """Find owned tiles adjacent to the path that can contribute army.

    Returns list of (tile, army, path_index) sorted by army descending.
    path_index is the index into `path` of the nearest path tile.
    """
    path_set = set(path)
    candidates: list[tuple[int, int, int]] = []

    for pi, cell in enumerate(path):
        r, c = divmod(cell, W)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < H and 0 <= nc < W:
                nb = nr * W + nc
                if nb in path_set:
                    continue
                if own_flat[nb] != my_slot:
                    continue
                a = int(arm_flat[nb])
                if a < 2:
                    continue
                candidates.append((nb, a, pi))
                path_set.add(nb)  # don't add the same tile twice

    candidates.sort(key=lambda x: -x[1])
    return candidates


def _build_move_sequence(
    path: list[int],
    side_gathers: list[tuple[int, int, int]],
    gen_flat: int,
    use_split: bool,
) -> list[tuple[int, int, int]]:
    """Build the flat move list: side-gathers first, then sweep."""
    moves: list[tuple[int, int, int]] = []

    # side-gathers ordered by injection point (source→target direction)
    sg_sorted = sorted(side_gathers, key=lambda x: x[2])
    for tile, _army, pi in sg_sorted:
        moves.append((tile, path[pi], 0))

    # sweep: roll army from source to target
    for i in range(len(path) - 1):
        is50 = 1 if (path[i] == gen_flat and use_split) else 0
        moves.append((path[i], path[i + 1], is50))

    return moves


def gather_path(
    view: PlayerView,
    cfg: BotConfig,
    target: int,
    max_moves: int | None = None,
    min_army: int | None = None,
    bypass_reserve: bool = False,
) -> GatherResult | None:
    own_flat = view.own.reshape(-1)
    arm_flat = view.arm.reshape(-1)
    H, W = view.H, view.W

    # 1. reserve
    reserve = reserve_amount(view, cfg)

    # 2. source selection
    result = _select_source(view, cfg, target, max_moves, reserve, bypass_reserve)
    if result is None:
        return None
    source, use_split = result

    if source == target:
        return None

    # 3. BFS path
    path = bfs_path(source, target, view.passable, H, W)
    if not path:
        return None

    sweep_cost = len(path) - 1

    # 4. side-gathers
    if max_moves is not None:
        remaining = max_moves - sweep_cost
    else:
        remaining = len(path) * 10  # generous cap for kill-shot
    max_side = min(remaining, int(cfg.SIDE_GATHER_FRACTION * len(path)))
    candidates = _find_side_gather_candidates(path, own_flat, arm_flat, view.my_slot, H, W)
    side_gathers = candidates[:max_side]

    # 5. army estimate — only owned tiles contribute
    side_sources = [sg[0] for sg in side_gathers]
    movers = [
        t for t in path[:-1]
        if own_flat[t] == view.my_slot and (t != view.gen_flat or not use_split)
    ]
    movers.extend(side_sources)
    arriving = mobile_army(movers, arm_flat, own_flat, view.my_slot)
    if use_split and view.gen_flat in path[:-1]:
        arriving += int(arm_flat[view.gen_flat]) // 2
    # target doesn't move — its army joins the stack if we own it
    target_tile = path[-1]
    if own_flat[target_tile] == view.my_slot:
        arriving += int(arm_flat[target_tile])

    if min_army is not None and arriving < min_army:
        return None

    # 6. build moves
    moves = _build_move_sequence(path, side_gathers, view.gen_flat, use_split)
    return GatherResult(moves=moves, arriving_army=arriving)
