"""Pure spatial / slot-ordering helpers for the obs encoder.

Low-level, dependency-free building blocks: Moore-neighborhood dilation, the
perspective→slot canonicalization pin, general-slot padding, and the
perspective-filtered board view. No `MemoryState` or `ObsConfig` dependency, so
this sits at the base of the obs package's import DAG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from game_types.state_constants import OWN_FOG


def _moore_dilate(mask: np.ndarray) -> np.ndarray:
    """Moore-neighborhood (3x3) binary dilation of a 2D bool mask.

    Same 8-direction OR pattern as `visibility.compute_visibility`, used by
    `step_memory` when expanding the agent's vision of an opponent's tile
    into "opponent had vision around here."
    """
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    return out


def canonical_slot_order(perspective_slot: int, P: int = 8) -> list[int]:
    """
    Raw slot ids in canonical channel order. Channel 0 is always the perspective player.

    See the module docstring of `obs` (Phase 1) for the full rationale —
    in brief, this is the standard slot-canonicalization trick that turns K
    per-game perspectives into K augmented views of the same dynamics rather
    than K decorrelated input distributions. Ascending-skip ordering for
    opponents matches the non-cyclic-game literature (Dota/AlphaStar/Pluribus).

    Example for P=8:
      perspective_slot=0 → [0, 1, 2, 3, 4, 5, 6, 7]
      perspective_slot=5 → [5, 0, 1, 2, 3, 4, 6, 7]
    """
    return [perspective_slot] + [s for s in range(P) if s != perspective_slot]


def pad_initial_generals(
    initial_generals, perspective_slot: int, num_slots: int = 8,
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


@dataclass(frozen=True)
class PerspectiveView:
    """A single-tick board state filtered to what the perspective observed.

    Cells outside the perspective's vision at this tick have `own = OWN_FOG`
    and `armies = 0` — matching what the perspective would see in the live
    game under fog of war. Cells within vision carry their true sim values.

    Used by `step_memory` to populate the dense-history buffer; the same
    abstraction is available for any future consumer that needs a per-tick
    "what the perspective sees" snapshot (e.g., a future pre-filter at the
    top of `build_obs` for current-tick cats).
    """
    own: np.ndarray    # int8 [H, W]
    armies: np.ndarray  # int16 [H, W]


def make_perspective_view(
    own_raw: np.ndarray, armies_raw: np.ndarray, vis: np.ndarray,
) -> PerspectiveView:
    """Build a `PerspectiveView` from raw sim arrays + the perspective's
    current vision mask. Fog cells get the OWN_FOG sentinel and zero armies.
    """
    return PerspectiveView(
        own=np.where(vis, own_raw, OWN_FOG).astype(np.int8),
        armies=np.where(vis, armies_raw, 0).astype(np.int16),
    )
