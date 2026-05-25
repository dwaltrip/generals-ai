"""Bot world model — thin wrapper over MemoryState for eval-bot-specific queries.

Milestone 1: just the known-passable mask for BFS. Future milestones add
ThreatWindow, ContactSet reads, scoring, etc.
"""

from __future__ import annotations

import numpy as np

# TODO: cross-package dep on training internals — extract shared
# fog-tracking module if this coupling becomes painful.
from bc.obs import MemoryState


def known_passable_mask(
    mem: MemoryState, H: int, W: int,
) -> np.ndarray:
    """Flat bool [H*W] passability mask for the bot's BFS.

    v1 policy (design doc §7.1): structures in fog are treated as
    impassable (mountain vs. city is indistinguishable behind fog).
    No city-traversability ratio — simpler than the NN's
    compute_known_passable.
    """
    structures_in_fog = (
        mem.is_structure
        & ~mem.known_mountain
        & ~mem.known_city
        & ~mem.known_general
    )
    impassable = mem.known_mountain | structures_in_fog
    return (~impassable).reshape(H * W)
