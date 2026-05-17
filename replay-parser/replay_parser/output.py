"""Per-game intermediate output writers.

Two sibling files per game (see `docs/2026-05/5.16-3-parser-output-design.md`):
    <id>.npz       — sim output (frozen alongside the sim)
    <id>.meta.npz  — per-perspective metadata (iterates independently)

`write_sim_output` is sim-aware only: it takes the finished `sim_core.State`
plus the decoded `ReplayData` and knows nothing about perspectives or DB.
`write_metadata` is a thin np.savez_compressed wrapper.
"""
from pathlib import Path

import numpy as np

from replay_parser.decode import ReplayData
import sim_core


def write_sim_output(state: sim_core.State, replay: ReplayData, out_path: Path) -> None:
    static = replay.static
    snaps_own = state.snapshots_ownership
    snaps_arm = state.snapshots_armies

    payload = {
        "replay_id": np.asarray(static.id, dtype="<U16"),
        "version": np.asarray(static.version, dtype=np.int32),
        "map_width": np.asarray(static.map_width, dtype=np.int32),
        "map_height": np.asarray(static.map_height, dtype=np.int32),
        "mountains": np.asarray(static.mountains, dtype=np.int32),
        "initial_cities": np.asarray(static.initial_cities, dtype=np.int32),
        "initial_city_armies": np.asarray(static.initial_city_armies, dtype=np.int32),
        "initial_neutrals": np.asarray(static.initial_neutrals, dtype=np.int32),
        "initial_neutral_armies": np.asarray(static.initial_neutral_armies, dtype=np.int32),
        "initial_generals": np.asarray(static.initial_generals, dtype=np.int32),
        "ownership": np.stack(snaps_own, axis=0).astype(np.int8, copy=False),
        "armies": np.stack(snaps_arm, axis=0).astype(np.int16, copy=False),
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


def write_metadata(metadata: dict[str, np.ndarray], out_path: Path) -> None:
    # pyright: ignore — pyright treats `**metadata` as if a key could be `allow_pickle`
    # (bool), conflicting with ndarray values. False positive on the dict spread.
    np.savez_compressed(out_path, **metadata)  # pyright: ignore[reportArgumentType]


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
