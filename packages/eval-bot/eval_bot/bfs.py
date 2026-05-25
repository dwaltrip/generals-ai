"""Integer-distance BFS over known-passable tiles.

4-connected (cardinal directions only — generals.io moves are orthogonal).
Returns raw hop counts, not the log-scaled distances the NN's bfs.py uses.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def bfs_distances(
    source: int, passable: np.ndarray, H: int, W: int,
) -> np.ndarray:
    """BFS from a single source cell.

    Args:
        source: flat cell index (row * W + col).
        passable: bool [H*W] — True where traversable.
        H, W: board dimensions.

    Returns:
        int32 [H*W]. Distance from source in hops. -1 for unreachable
        (impassable or disconnected).
    """
    HW = H * W
    dist = np.full(HW, -1, dtype=np.int32)
    if not passable[source]:
        return dist

    dist[source] = 0
    q: deque[int] = deque()
    q.append(source)

    while q:
        cell = q.popleft()
        d = dist[cell] + 1
        r, c = divmod(cell, W)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < H and 0 <= nc < W:
                nb = nr * W + nc
                if passable[nb] and dist[nb] == -1:
                    dist[nb] = d
                    q.append(nb)

    return dist


def bfs_path(
    source: int, target: int, passable: np.ndarray, H: int, W: int,
) -> list[int]:
    """Shortest path from source to target (inclusive of both endpoints).

    Returns an empty list if the target is unreachable.
    """
    if source == target:
        return [source]

    HW = H * W
    prev = np.full(HW, -1, dtype=np.int32)
    visited = np.zeros(HW, dtype=bool)

    if not passable[source] or not passable[target]:
        return []

    visited[source] = True
    q: deque[int] = deque()
    q.append(source)

    found = False
    while q:
        cell = q.popleft()
        r, c = divmod(cell, W)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < H and 0 <= nc < W:
                nb = nr * W + nc
                if passable[nb] and not visited[nb]:
                    visited[nb] = True
                    prev[nb] = cell
                    if nb == target:
                        found = True
                        break
                    q.append(nb)
        if found:
            break

    if not found:
        return []

    # reconstruct
    path = []
    cell = target
    while cell != -1:
        path.append(cell)
        cell = int(prev[cell])
    path.reverse()
    return path
