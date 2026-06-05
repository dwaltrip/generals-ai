"""
BFS distance-from-known-generals with per-source caching.

The obs tensor includes 8 channels of "distance from the perspective to each
general" (self + 7 opponents). Each channel is the log-scaled shortest-path
distance on the perspective's *known-passable* graph (see §"Knobs" below for
what counts as passable).

--- Cache strategy ---

The known-passable graph grows monotonically across a (game, perspective)
walk — once a cell becomes known-passable, it never reverts. The graph
changes when:
  (a) A `structures_in_fog` cell is resolved to a city (visible city tile).
  (b) A `structures_in_fog` cell is resolved to a general (visible general).

Both are rare events — ~tens per game across ~600 frames. Most frames see
*zero* graph changes, and most general sources don't move once observed
(generals are static). So aggressive caching collapses BFS cost to ~10% of
the uncached baseline.

Per-general cache: each general (0=self, 1..7=canonical opp) has its own
(graph_epoch, source, distances) entry. We bump `cache.graph_epoch` when
the graph grows; a per-entry mismatch on either epoch or source triggers
recomputation. Sources for opp generals are `-1` (unknown) until first
sighted; the channel is filled with the -1 sentinel during that window.

--- Knobs (set in caller, e.g. obs.py) ---

Several BFS-policy decisions are *not* baked into this module — they're
encoded in the `known_passable` mask the caller hands in. Current v1
spike policy (see `obs.py` cat 5):

  - Enemy generals: impassable. Own general passable.
  - Cities: own always passable. Non-own (neutral / enemy) passable iff
    `total_army > city_army * CITY_TRAVERSABILITY_FACTOR` — crude
    "do I have enough army to take it" heuristic, scales with both
    perspective strength and defender strength.
  - `structures_in_fog`: impassable (agent plans on what it knows).

Post-spike upgrade is weighted-edge BFS (Dijkstra) with per-cell costs
reflecting actual capture cost. Until then, the per-frame passability
mask is the only lever.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Sentinel used in the raw integer distance array for unreachable cells.
# After log-scaling in `_to_channel`, this becomes -1.0 in the obs channel.
UNREACHABLE = -1


@dataclass
class _Entry:
    cached_graph_epoch: int = -1
    cached_source: int = -1
    distances: np.ndarray | None = None  # float32 [H, W], log-scaled with -1 sentinel


@dataclass
class BFSCache:
    """Per-(game, perspective) BFS cache. Caller initializes once per (game, k)."""

    P: int = 8
    graph_epoch: int = 0
    # Index 0 = self general; 1..7 = canonical opp generals (matches obs channel layout).
    per_general: list[_Entry] = field(default_factory=list)
    # Mask cached by `maybe_invalidate` for frame-to-frame equality checks.
    # Held by reference (compute_known_passable allocates fresh each frame).
    prev_mask: np.ndarray | None = field(default=None, repr=False)
    # Per-cache lookup accounting. Incremented in `compute_or_get`. Excludes
    # the source<0 early-return (those calls never touch the cache). `n_lookups`
    # is the denominator; `n_bfs_runs` is the number of actual `_bfs` calls.
    # Hit rate = 1 - n_bfs_runs / n_lookups.
    n_lookups: int = 0
    n_bfs_runs: int = 0

    def __post_init__(self) -> None:
        if not self.per_general:
            self.per_general = [_Entry() for _ in range(self.P)]

    def invalidate_graph(self) -> None:
        """Bump the epoch so all per-general entries miss on next lookup."""
        self.graph_epoch += 1

    def maybe_invalidate(self, mask: np.ndarray) -> None:
        """Bump the epoch iff `mask` differs from the previously cached one.

        One `np.array_equal` per frame (~µs on a ~1024-cell bool) replaces the
        unconditional per-frame invalidation that used to live in the dataset
        walk. Both invalidation triggers are covered here in a single check:
        structural growth (new known-passable cells from `step_memory`) and
        city-passability flips (per-tick army-ratio changes). The mask is the
        single source of truth for "would BFS produce different output?"
        """
        if self.prev_mask is None or not np.array_equal(self.prev_mask, mask):
            self.graph_epoch += 1
            self.prev_mask = mask


def init_bfs_cache(P: int = 8) -> BFSCache:
    return BFSCache(P=P)


def compute_or_get(
    cache: BFSCache,
    gen_idx: int,
    source: int,
    known_passable_flat: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """
    Get (or compute and cache) the BFS distance channel for one general.

    Args:
        cache: per-(game, perspective) BFSCache.
        gen_idx: 0 for self general, 1..P-1 for canonical opponents.
        source: flat cell index of this general (unpadded coord space, i.e.
            `row * W + col`). Pass `-1` if this general's location is not yet
            known — returns an all-sentinel channel without computing.
        known_passable_flat: bool [H*W] — cells the perspective considers
            traversable. Mountains and `structures_in_fog` cells are False;
            everything else (including cities + generals at v1 uniform cost)
            is True.
        H, W: unpadded board dims.

    Returns:
        float32 [H, W] log-scaled distance, `-1.0` for unreachable / unrevealed.
    """
    if source < 0:
        return np.full((H, W), -1.0, dtype=np.float32)

    cache.n_lookups += 1
    entry = cache.per_general[gen_idx]
    if (
        entry.distances is not None
        and entry.cached_graph_epoch == cache.graph_epoch
        and entry.cached_source == source
    ):
        return entry.distances

    cache.n_bfs_runs += 1
    distances = _bfs(source, known_passable_flat, H, W)
    entry.cached_graph_epoch = cache.graph_epoch
    entry.cached_source = source
    entry.distances = distances
    return distances


def _log_scale_dist(dist_2d: np.ndarray) -> np.ndarray:
    """Log-scale a raw integer BFS-distance array, preserving the -1 sentinel
    for unreachable cells.

    `np.maximum(x, 0)` before log1p suppresses the "log of negative" warning
    for cells where the -1 sentinel will be substituted by np.where anyway.
    """
    dist_f = dist_2d.astype(np.float32)
    safe = np.maximum(dist_f, 0)
    return np.where(dist_f >= 0, np.log1p(safe), -1.0).astype(np.float32)


def _bfs(source: int, known_passable_flat: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Layered 4-connected BFS — one numpy wavefront per distance layer.

    Each iteration expands the current frontier in all four cardinal
    directions at once via boundary-aware slice assignment (same pattern as
    `visibility.compute_visibility`'s Moore expansion, restricted to
    4-connected). Loop count is O(diameter) instead of O(cells), so the
    Python interpreter tax is per-layer, not per-cell.

    `dist[source] = 0` unconditionally — even if the source cell isn't itself
    in `known_passable_flat`. The wavefront only propagates *into* passable
    cells, so neighbors of a non-passable source are still reached correctly
    (matches the structures-in-fog edge case where a general's cell briefly
    sits in `structures_in_fog` during early-game initialization).

    Returns log-scaled float32 [H, W] distance array. Unreachable cells are
    encoded as -1.0 (post-log sentinel — never feed -1 into log1p).
    """
    passable_2d = known_passable_flat.reshape(H, W)
    dist = np.full((H, W), UNREACHABLE, dtype=np.int32)

    sr, sc = divmod(source, W)
    dist[sr, sc] = 0

    visited = np.zeros((H, W), dtype=bool)
    visited[sr, sc] = True
    frontier = np.zeros((H, W), dtype=bool)
    frontier[sr, sc] = True

    d = 0
    while True:
        # 4-connected dilation of `frontier`. Each line shifts the frontier
        # one step in one cardinal direction and ORs it into the neighbor
        # mask — "if (r+1, c) was in frontier, then (r, c) is its N neighbor",
        # etc. Boundary slices drop cells whose neighbor would be off-board.
        neighbors = np.zeros((H, W), dtype=bool)
        neighbors[:-1, :] = frontier[1:, :]    # N (frontier-cell is south of us)
        neighbors[1:, :] |= frontier[:-1, :]   # S
        neighbors[:, :-1] |= frontier[:, 1:]   # W
        neighbors[:, 1:] |= frontier[:, :-1]   # E

        new_layer = neighbors & passable_2d & ~visited
        if not new_layer.any():
            break

        d += 1
        dist[new_layer] = d
        visited |= new_layer
        frontier = new_layer

    return _log_scale_dist(dist)
