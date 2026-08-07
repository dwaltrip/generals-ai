"""Pure spatial helpers for the obs encoder.

Low-level, dependency-free building blocks: Moore-neighborhood dilation and
the perspective-filtered board view. No `MemoryState` or `ObsConfig`
dependency, so this sits at the base of the obs package's import DAG.
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
