"""Helpers for adapting sim_core structures to the dict shape that
bc.obs.build_obs and bc.mask.build_mask consume.
"""

from __future__ import annotations

import numpy as np


def _capture_events_to_array(events) -> np.ndarray:
    """Pack capture events into the `[N, 3] int32` shape (timestep, captor,
    captured) that step_memory's filter `events[events[:, 0] == t]` expects.

    Returns shape `(0, 3)` for empty event lists so `.size > 0` is False
    and the downstream branch is correctly skipped.
    """
    if not events:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(
        [[e.timestep, e.captor, e.captured] for e in events],
        dtype=np.int32,
    )
