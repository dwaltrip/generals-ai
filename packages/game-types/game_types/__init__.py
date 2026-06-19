"""Data contracts that cross the game/model boundary.

These types let the model-inference code (`training/bc`) accept a *typed*
observation without importing the game engine (`sim_core`), and let the
generic driver (`game-runner`) hand observations to a model without importing
the model package. They are deliberately dependency-light (numpy only) so both
sides can depend on them without dragging in each other's deps.

  - StaticMap   — the unchanging board spec (dims, mountains, initial cities /
                  generals / neutrals) that `sim_core.new_state` consumes. The
                  game side reads it directly; the replay parser produces it
                  inside its richer `GameStatic` record.
  - PlayerView  — produced by `game_runner.state_to_view`, consumed by the
                  model encoder. One perspective's raw game state at tick t.
  - ObsBundle   — encoder output: the model input tensor + masks.
  - Decision    — decoder output: the chosen wire move + raw diagnostics.

PlayerView carries the *raw* (un-fogged) per-tick arrays the inference code
reads off `sim_core.State`. Fog-of-war and the 8-slot general padding are
applied model-side in the encoder. `initial_generals` is the raw per-player
array (length = num_players); the encoder pads it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class StaticMap:
    """A game's static map spec — the unchanging inputs `sim_core.new_state`
    consumes to build a board, and that the runner / agents / save / viewer
    read while a game plays out.

    Tile arrays are flat cell indices in unpadded `row*W + col` order.
    `initial_generals` is the raw per-player array (length = num_players),
    with a `-1` sentinel for any null slot.
    """

    map_width: int
    map_height: int
    usernames: list[str]
    mountains: list[int]                # flat cell indices
    initial_cities: list[int]           # flat cell indices
    initial_city_armies: list[int]
    initial_generals: list[int]         # flat cell index per player (-1 = null slot)
    initial_neutrals: list[int]         # flat cell indices
    initial_neutral_armies: list[int]


@dataclass(frozen=True, eq=False)
class PlayerView:
    """One perspective's raw observation of the game at a single tick.

    Index spaces follow the sim: ownership/armies are flat `[H*W]` in unpadded
    (row*W + col) order; the `*_cells` / `mountains` arrays are flat cell
    indices; `capture_events` is `[N, 3]` rows of `(timestep, captor, captured)`.
    """

    slot: int                       # perspective player slot
    timestep: int
    H: int
    W: int
    ownership_t: np.ndarray         # int8  [H*W]  (-2=mountain, -1=neutral, 0..7=slot)
    armies_t: np.ndarray            # int16 [H*W]
    mountains: np.ndarray           # int32 flat cell indices (static)
    initial_cities: np.ndarray      # int32 flat cell indices (static)
    initial_generals: np.ndarray    # int32 flat cell index per player (raw, len = num_players)
    cities: np.ndarray              # int32 flat cell indices (dynamic; grows mid-game)
    cities_present_at: np.ndarray   # int32 first tick each city existed
    capture_events: np.ndarray      # int32 [N, 3]; shape (0, 3) when empty
    # Death (surrender / capture-while-alive) and neutralize events — board-wide
    # announcements a live player sees, fog or not (like captures). Drive the
    # obs player-status channels. Both `[N, 2]` rows of (timestep, player);
    # shape (0, 2) when empty.
    death_events: np.ndarray        # int32 [N, 2]
    neutralize_events: np.ndarray   # int32 [N, 2]


@dataclass(frozen=True, eq=False)
class ObsBundle:
    """Encoder output for one perspective — the model's per-sample input."""

    obs: np.ndarray                 # float32 [OBS_CHANNELS, H_PADDED, W_PADDED]
    policy_mask: np.ndarray         # bool    [H_PADDED, W_PADDED, 8]  (per-cell legality)
    valid_mask: np.ndarray          # bool    [1, H_PADDED, W_PADDED]  (unpadded board region)


@dataclass(frozen=True)
class Decision:
    """Decoder output for one perspective."""

    move: tuple[int, int, int]      # wire (source, dest, is50); (-1, -1, -1) = pass
    category: str                   # "moved" | "passed" | "no_legal"
    diag: dict                      # raw per-tick model-introspection stats
