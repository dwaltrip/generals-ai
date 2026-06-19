"""Adapt `sim_core` structures into the neutral data contracts that model
inference consumes.

`state_to_view` is the single place that reads `sim_core.State` fields for a
perspective and packs them into a `game_types.PlayerView` — the boundary that
keeps the model package free of any `sim_core` dependency.
"""

from __future__ import annotations

import numpy as np

from game_types import PlayerView, StaticMap
import sim_core


def state_to_view(state: sim_core.State, map_data: StaticMap, slot: int) -> PlayerView:
    """Extract one perspective's raw observation of the current tick.

    Reads exactly the `State` / `StaticMap` fields the model encoder needs; fog
    and the 8-slot general padding are applied downstream in the encoder, so the
    view carries the raw board arrays.
    """
    t = state.timestep
    return PlayerView(
        slot=slot,
        timestep=t,
        H=map_data.map_height,
        W=map_data.map_width,
        ownership_t=np.asarray(state.snapshots_ownership[t], dtype=np.int8),
        armies_t=np.asarray(state.snapshots_armies[t], dtype=np.int16),
        mountains=np.asarray(map_data.mountains, dtype=np.int32),
        initial_cities=np.asarray(map_data.initial_cities, dtype=np.int32),
        initial_generals=np.asarray(map_data.initial_generals, dtype=np.int32),
        cities=np.asarray(state.cities, dtype=np.int32),
        cities_present_at=np.asarray(state.cities_present_at, dtype=np.int32),
        capture_events=capture_events_to_array(state.capture_events),
        death_events=player_events_to_array(state.death_events),
        neutralize_events=player_events_to_array(state.neutralize_events),
    )


def capture_events_to_array(events) -> np.ndarray:
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


def player_events_to_array(events) -> np.ndarray:
    """Pack death / neutralize events into `[N, 2] int32` rows of (timestep,
    player) — the shape `precompute_player_status` reads. Returns `(0, 2)` for
    empty lists."""
    if not events:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array([[e.timestep, e.player] for e in events], dtype=np.int32)
