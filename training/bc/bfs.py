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

from collections import deque
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

    def __post_init__(self) -> None:
        if not self.per_general:
            self.per_general = [_Entry() for _ in range(self.P)]

    def invalidate_graph(self) -> None:
        """Bump the epoch so all per-general entries miss on next lookup."""
        self.graph_epoch += 1


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

    entry = cache.per_general[gen_idx]
    if (
        entry.distances is not None
        and entry.cached_graph_epoch == cache.graph_epoch
        and entry.cached_source == source
    ):
        return entry.distances

    distances = _bfs(source, known_passable_flat, H, W)
    entry.cached_graph_epoch = cache.graph_epoch
    entry.cached_source = source
    entry.distances = distances
    return distances


def _bfs(source: int, known_passable_flat: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Plain 4-connected BFS on a [H*W] passable mask.

    Returns log-scaled float32 [H, W] distance array. Unreachable cells are
    encoded as -1.0 (post-log sentinel — never feed -1 into log1p).

    Implementation note: Python-level BFS loop is fine at our scale (~1024
    cells, ~tens of recomputes per game thanks to caching). A vectorized
    bellman-ford or scipy.sparse.csgraph.shortest_path would be faster but
    adds a dependency for ~no measurable benefit when cache hit rate is
    90%+ over the full corpus.
    """
    HW = H * W
    dist = np.full(HW, UNREACHABLE, dtype=np.int32)

    # Walk from the source if it's itself passable; otherwise start the
    # frontier as the immediate passable neighbors (handles the case where the
    # general's cell is the source even though structures_in_fog might briefly
    # mask it during early-game initialization). Generals are passable in v1
    # so source is normally in the passable set.
    dist[source] = 0
    frontier: deque[int] = deque([source])

    while frontier:
        cell = frontier.popleft()
        d = dist[cell]
        r, c = divmod(cell, W)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            ncell = nr * W + nc
            if known_passable_flat[ncell] and dist[ncell] == UNREACHABLE:
                dist[ncell] = d + 1
                frontier.append(ncell)

    # Log-scale + sentinel passthrough. `np.maximum(x, 0)` before log1p suppresses
    # the "log of negative" warning for cells where the -1 sentinel will be
    # substituted by np.where anyway.
    dist_f = dist.astype(np.float32).reshape(H, W)
    safe = np.maximum(dist_f, 0)
    return np.where(dist_f >= 0, np.log1p(safe), -1.0).astype(np.float32)
