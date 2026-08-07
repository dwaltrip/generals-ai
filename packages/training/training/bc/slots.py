"""Slot-ordering and slot-capacity helpers shared by the training pipeline and the live path.

Channel layouts and per-player target arrays are sized for a fixed
`MAX_PLAYERS` slots. This module owns that constant, the canonical
slot→channel ordering, and the padding that adapts smaller games to the
fixed layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Slot capacity of the encoder's channel layout and the per-player arrays.
# The corpus filter (`constants.ELIGIBLE_PLAYER_COUNT`) is a separate,
# policy-level constant for the exact player count required of training games.
MAX_PLAYERS = 8


@dataclass(frozen=True)
class SlotOrder:
    """A perspective's canonical slot ordering: its own slot first, then the
    opponents in ascending slot order. `order[i]` is the raw slot id at
    position `i` of any per-player layout (per-opp obs channel groups,
    `[MAX_PLAYERS]` target vectors).

    Example for P=8:
      perspective 0 → (0, 1, 2, 3, 4, 5, 6, 7)
      perspective 5 → (5, 0, 1, 2, 3, 4, 6, 7)

    Self-first ordering is the standard canonicalization for multi-agent inputs.
    Prior art: Dota/AlphaStar/Pluribus do the same.
    The naive alternative is channels laid out by raw slot id. This puts the model's
    "self" view in a different channel for each perspective, and the model would have
    to re-learn the same patterns once for each slot.
    With "self" pinned to channel 0, channel meanings are consistent regardless of
    which raw slot the perspective occupies.
    The game's K perspectives become K augmented views of one input distribution.
    """

    order: tuple[int, ...]

    @classmethod
    def for_perspective(cls, perspective_slot: int, P: int = MAX_PLAYERS) -> SlotOrder:
        return cls((perspective_slot, *(s for s in range(P) if s != perspective_slot)))

    @property
    def perspective(self) -> int:
        return self.order[0]

    @property
    def opp_slots(self) -> list[int]:
        return list(self.order[1:])


def pad_initial_generals(
    initial_generals, perspective_slot: int, num_slots: int = MAX_PLAYERS,
) -> np.ndarray:
    """Pad `initial_generals` to `num_slots` length for the obs encoder.

    The BC encoder indexes slots 0..num_slots-1 unconditionally. For
    <num_slots-player games, unused slots are filled with the perspective's own
    general cell — idempotent under fancy indexing (sets an already-set
    own-general bit) and safe in step_memory's per-player loop (the own general
    is known from t=0, so padded slots never trigger false sightings).
    """
    ig = np.asarray(initial_generals, dtype=np.int32)
    if len(ig) >= num_slots:
        return ig
    own = int(ig[perspective_slot])
    padded = np.full(num_slots, own, dtype=np.int32)
    padded[: len(ig)] = ig
    return padded
