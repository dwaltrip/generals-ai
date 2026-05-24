"""Adapt a live `sim_core.State` to the dict shape that bc.obs.build_obs
and bc.mask.build_mask consume.

The training-time observation encoder reads a `sim` dict whose
`ownership` / `armies` fields are stacked `[T, HW]` ndarrays for the
whole game. Online inference doesn't have the full game in hand — it
has a State that grows one snapshot per tick. The encoder's read
surface, however, is narrow:

  - `sim["ownership"][t]` and `sim["armies"][t]` — current snapshot only
    (history is carried in MemoryState, not in `sim`)
  - `sim["ownership"].shape[0]` — used as `max_turns` for the
    game-progress channel in `_cat_self_broadcast`

So the time-indexed fields are wrapped in `_TimeIndexedArray`, a tiny
duck-type that forwards `[t]` to the live snapshot list and reports a
configurable `max_turns_hint` for `.shape[0]`. Static fields and event
lists pass through unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import sim_core


class _TimeIndexedArray:
    """View over State.snapshots_* satisfying the read shape `build_obs`
    expects of `sim["ownership"]` / `sim["armies"]`:

      - `view[t]` returns the snapshot at t (a 1D ndarray, ready for
        `.reshape(H, W)`)
      - `view.shape[0]` returns the configured `max_turns` hint

    `max_turns` is decoupled from the actual snapshot count because the
    encoder uses it to compute `game_progress = t / max_turns`. If we
    let it equal the always-growing `snapshots_len`, `game_progress`
    would sit at ~1.0 throughout the game. A fixed upper-bound hint
    keeps the signal monotonic and in-distribution with training.
    """

    __slots__ = ("_snapshots", "_max_turns")

    def __init__(self, snapshots, max_turns_hint: int):
        self._snapshots = snapshots
        self._max_turns = max_turns_hint

    def __getitem__(self, t: int) -> np.ndarray:
        return self._snapshots[t]

    @property
    def shape(self) -> tuple[int]:
        return (self._max_turns,)


def sim_dict_from_state(
    state: sim_core.State,
    static: Any,
    max_turns_hint: int = 1000,
) -> dict[str, Any]:
    """Build the `sim` dict that bc.obs.build_obs / bc.mask.build_mask consume.

    `static` is the same record fed to `sim_core.new_state(...)` — supplies
    mountains, initial cities, and initial generals (none of which the
    State exposes directly; they're baked into the initial board).

    `max_turns_hint` is the game-length upper bound used by the encoder's
    game-progress channel. 1000 is a reasonable mean for 2-player FFA
    games on the corpus; tune later if the model behaves out-of-distribution.
    """
    return {
        "ownership": _TimeIndexedArray(state.snapshots_ownership, max_turns_hint),
        "armies": _TimeIndexedArray(state.snapshots_armies, max_turns_hint),
        "mountains": np.asarray(static.mountains, dtype=np.int32),
        "initial_cities": np.asarray(static.initial_cities, dtype=np.int32),
        "initial_generals": np.asarray(static.initial_generals, dtype=np.int32),
        "cities": np.asarray(state.cities, dtype=np.int32),
        "cities_present_at": np.asarray(state.cities_present_at, dtype=np.int32),
        "capture_events": _capture_events_to_array(state.capture_events),
    }


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
