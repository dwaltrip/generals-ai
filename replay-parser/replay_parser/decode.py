from dataclasses import dataclass

import numpy as np

from replay_parser._collector.wire import decode as _decompress
from replay_parser.types import TileIndex


# Wire field name mapping (per docs/replay-format.md and plan §3.2):
#   moves[].start / .end / .turn   →  source / dest / timestep
#   afks[].turn                    →  timestep
# All other internal field names match the wire format.


@dataclass(frozen=True, slots=True)
class Moves:
    index: np.ndarray     # int8[N]
    source: np.ndarray    # int16[N]
    dest: np.ndarray      # int16[N]
    is50: np.ndarray      # uint8[N]
    timestep: np.ndarray  # int32[N]


@dataclass(frozen=True, slots=True)
class Afks:
    index: np.ndarray     # int8[N]
    timestep: np.ndarray  # int32[N]


@dataclass(frozen=True, slots=True)
class GameStatic:
    id: str
    version: int
    map_width: int
    map_height: int
    usernames: list[str]
    stars: list[float]
    initial_cities: list[TileIndex]
    initial_city_armies: list[int]
    initial_generals: list[TileIndex]       # -1 sentinel for any null slot (empirically never in filtered FFA)
    mountains: list[TileIndex]
    initial_neutrals: list[TileIndex]
    initial_neutral_armies: list[int]
    modifiers: list[int]


@dataclass(frozen=True, slots=True)
class ReplayData:
    static: GameStatic
    moves: Moves
    afks: Afks


def decode_wire(raw: bytes) -> ReplayData:
    return decode_wire_array(_decompress(raw))


def decode_wire_array(wire: list) -> ReplayData:
    return ReplayData(
        static=_decode_static(wire),
        moves=_decode_moves(wire[10]),
        afks=_decode_afks(wire[11]),
    )


def _decode_static(wire: list) -> GameStatic:
    return GameStatic(
        version=wire[0],
        id=wire[1],
        map_width=wire[2],
        map_height=wire[3],
        usernames=list(wire[4]),
        stars=list(wire[5]),
        initial_cities=list(wire[6]),
        initial_city_armies=list(wire[7]),
        initial_generals=[g if g is not None else -1 for g in wire[8]],
        mountains=list(wire[9]),
        initial_neutrals=list(wire[14]),
        initial_neutral_armies=list(wire[15]),
        modifiers=list(wire[21]),
    )


def _decode_moves(wire_moves: list) -> Moves:
    if not wire_moves:
        return Moves(
            index=np.empty(0, dtype=np.int8),
            source=np.empty(0, dtype=np.int16),
            dest=np.empty(0, dtype=np.int16),
            is50=np.empty(0, dtype=np.uint8),
            timestep=np.empty(0, dtype=np.int32),
        )
    arr = np.asarray(wire_moves, dtype=np.int64)
    return Moves(
        index=arr[:, 0].astype(np.int8),
        source=arr[:, 1].astype(np.int16),
        dest=arr[:, 2].astype(np.int16),
        is50=arr[:, 3].astype(np.uint8),
        timestep=arr[:, 4].astype(np.int32),
    )


def _decode_afks(wire_afks: list) -> Afks:
    if not wire_afks:
        return Afks(
            index=np.empty(0, dtype=np.int8),
            timestep=np.empty(0, dtype=np.int32),
        )
    arr = np.asarray(wire_afks, dtype=np.int64)
    return Afks(
        index=arr[:, 0].astype(np.int8),
        timestep=arr[:, 1].astype(np.int32),
    )
