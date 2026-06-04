"""Save an eval game to a compressed .npz file.

Modeled on replay_parser.output.write_sim_output but takes a `StaticMap`
(from seed_map) instead of a ReplayData. Produces the same array layout
so existing viewers/tools can read the output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from game_types import StaticMap
import sim_core


def write_eval_game(state: sim_core.State, map_data: StaticMap, out_path: Path) -> None:
    payload = {
        "map_width": np.asarray(map_data.map_width, dtype=np.int32),
        "map_height": np.asarray(map_data.map_height, dtype=np.int32),
        "mountains": np.asarray(map_data.mountains, dtype=np.int32),
        "initial_cities": np.asarray(map_data.initial_cities, dtype=np.int32),
        "initial_city_armies": np.asarray(map_data.initial_city_armies, dtype=np.int32),
        "initial_neutrals": np.asarray(map_data.initial_neutrals, dtype=np.int32),
        "initial_neutral_armies": np.asarray(map_data.initial_neutral_armies, dtype=np.int32),
        "initial_generals": np.asarray(map_data.initial_generals, dtype=np.int32),
        "ownership": np.stack(state.snapshots_ownership, axis=0).astype(np.int8, copy=False),
        "armies": np.stack(state.snapshots_armies, axis=0).astype(np.int16, copy=False),
        "cities": np.asarray(state.cities, dtype=np.int32),
        "cities_present_at": np.asarray(state.cities_present_at, dtype=np.int32),
        "death_events": _pack_events_2(state.death_events),
        "capture_events": _pack_events_3(state.capture_events, "captor", "captured"),
        "neutralize_events": _pack_events_2(state.neutralize_events),
        "actions_source": np.asarray(state.actions_source, dtype=np.int16),
        "actions_dest": np.asarray(state.actions_dest, dtype=np.int16),
        "actions_is50": np.asarray(state.actions_is50, dtype=np.int8),
    }
    np.savez_compressed(out_path, **payload)


def _pack_events_2(events) -> np.ndarray:
    if not events:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array([(e.timestep, e.player) for e in events], dtype=np.int32)


def _pack_events_3(events, attr_a: str, attr_b: str) -> np.ndarray:
    if not events:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(
        [(e.timestep, getattr(e, attr_a), getattr(e, attr_b)) for e in events],
        dtype=np.int32,
    )
