"""ExpandOrExplore gate — the bot's default mode.

Two sub-modes switched by the ExpandableTiles / UnoccupiedFrontier ratio:
  - Expand: thin BFS crawl into adjacent empty tiles (cheap, no gather).
  - Explore: push the highest-army frontier tile toward the least-explored
    sector (needs BFS, no gather_tree yet in milestone 1).

Returns a (source, dest, is50) move tuple, or None when fully boxed in.
"""

from __future__ import annotations

import numpy as np

from bc.obs import MemoryState

from eval_bot.bfs import bfs_distances

NEUTRAL = -1


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


def _pick_explore_move(
    own: np.ndarray,
    arm: np.ndarray,
    my_slot: int,
    mem: MemoryState,
    passable: np.ndarray,
    gen_flat: int,
    H: int,
    W: int,
) -> tuple[int, int, int] | None:
    """Push toward the least-explored sector. No gather_tree — just move
    the highest-army owned tile one step toward a fog target."""
    gen_r, gen_c = divmod(gen_flat, W)

    # score each sector by count of passable fog tiles
    fog_passable = ~mem.is_structure & ~mem.historically_seen
    sector_scores = [0, 0, 0, 0]
    fog_tiles_by_sector: list[list[int]] = [[], [], [], []]

    passable_2d = passable.reshape(H, W)
    for r in range(H):
        for c in range(W):
            if fog_passable[r, c] and passable_2d[r, c]:
                s = _sector_of(r - gen_r, c - gen_c)
                sector_scores[s] += 1
                fog_tiles_by_sector[s].append(r * W + c)

    if max(sector_scores) == 0:
        return None

    best_sector = int(np.argmax(sector_scores))
    fog_cells = fog_tiles_by_sector[best_sector]
    if not fog_cells:
        return None

    # source = any owned tile with army >= 2, best = highest army
    mine_flat = (own == my_slot).reshape(-1)
    arm_flat = arm.reshape(-1)
    candidates = np.where(mine_flat & (arm_flat >= 2))[0]
    if len(candidates) == 0:
        return None

    best_src = int(candidates[np.argmax(arm_flat[candidates])])

    # find the nearest fog tile in the winning sector via BFS from best_src
    dist_from_src = bfs_distances(best_src, passable, H, W)
    fog_set = set(fog_cells)
    nearest_fog = -1
    nearest_dist = H * W + 1
    for fc in fog_cells:
        d = dist_from_src[fc]
        if d >= 0 and d < nearest_dist:
            nearest_dist = d
            nearest_fog = fc

    if nearest_fog == -1:
        return None

    # move one step toward that fog tile: BFS from fog target, step downhill
    dist_from_target = bfs_distances(nearest_fog, passable, H, W)
    if dist_from_target[best_src] < 0:
        return None

    r, c = divmod(best_src, W)
    best_nb = -1
    best_d = dist_from_target[best_src]
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < H and 0 <= nc < W:
            nb = nr * W + nc
            d = dist_from_target[nb]
            if d >= 0 and d < best_d:
                best_d = d
                best_nb = nb

    if best_nb == -1:
        return None

    return (best_src, best_nb, 0)


EXPAND_EXPLORE_THRESHOLD = 0.4

MOVE_EXPAND = "expand"
MOVE_EXPLORE = "explore"

# Return type: (src, dst, is50, mode) or None
ExpandResult = tuple[int, int, int, str] | None


def pick_expand_or_explore(
    own: np.ndarray,
    arm: np.ndarray,
    my_slot: int,
    mem: MemoryState,
    passable: np.ndarray,
    gen_flat: int,
    H: int,
    W: int,
    expand_explore_threshold: float = EXPAND_EXPLORE_THRESHOLD,
) -> ExpandResult:
    """Top-level ExpandOrExplore gate. Returns (src, dst, is50, mode) or None."""
    expandable = _expandable_tiles(own, arm, my_slot, H, W)
    frontier = _unoccupied_frontier(own, my_slot, H, W)

    expandable_count = int(expandable.sum())
    frontier_count = int(frontier.sum())

    if frontier_count == 0:
        return None

    if expandable_count > 0 and expandable_count / frontier_count >= expand_explore_threshold:
        gen_dist = bfs_distances(gen_flat, passable, H, W)
        move = _pick_expand_move(expandable, own, arm, gen_dist, H, W)
        if move is not None:
            return (*move, MOVE_EXPAND)

    # explore mode: low ratio of expandable-to-frontier, or no expandable tiles
    move = _pick_explore_move(
        own, arm, my_slot, mem, passable, gen_flat, H, W,
    )
    if move is not None:
        return (*move, MOVE_EXPLORE)

    # explore couldn't find a move (e.g. no army >= 2 on frontier) — fall
    # back to expand if any expandable tiles exist
    if expandable_count > 0:
        gen_dist = bfs_distances(gen_flat, passable, H, W)
        move = _pick_expand_move(expandable, own, arm, gen_dist, H, W)
        if move is not None:
            return (*move, MOVE_EXPAND)

    return None
