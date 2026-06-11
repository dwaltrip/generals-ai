"""Load an eval game .npz back into its raw arrays.

The read-side counterpart to save.write_eval_game: this module owns the npz
key names and dtypes so callers don't each hard-code the schema. It returns
the arrays as-stored; shaping them into viewer state, GameRecords, etc. is
each caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EvalGameNpz:
    map_width: int
    map_height: int

    # static map
    mountains: np.ndarray
    initial_cities: np.ndarray
    initial_city_armies: np.ndarray
    initial_neutrals: np.ndarray
    initial_neutral_armies: np.ndarray
    initial_generals: np.ndarray

    # state evolution
    ownership: np.ndarray  # [T+1, H*W] int8
    armies: np.ndarray  # [T+1, H*W] int16
    cities: np.ndarray  # [C] tile indices
    cities_present_at: np.ndarray  # [C] first snapshot index

    # events, in the packed layout save.py writes
    death_events: np.ndarray  # [D, 2] (tick, player)
    capture_events: np.ndarray  # [K, 3] (tick, captor, captured)
    neutralize_events: np.ndarray  # [N, 2] (tick, player)

    # actions
    actions_source: np.ndarray
    actions_dest: np.ndarray
    actions_is50: np.ndarray

    @property
    def num_timesteps(self) -> int:
        return self.ownership.shape[0]


def load_eval_game(npz_path: Path) -> EvalGameNpz:
    with np.load(npz_path) as z:
        return EvalGameNpz(
            map_width=int(z["map_width"]),
            map_height=int(z["map_height"]),
            mountains=z["mountains"],
            initial_cities=z["initial_cities"],
            initial_city_armies=z["initial_city_armies"],
            initial_neutrals=z["initial_neutrals"],
            initial_neutral_armies=z["initial_neutral_armies"],
            initial_generals=z["initial_generals"],
            ownership=z["ownership"],
            armies=z["armies"],
            cities=z["cities"],
            cities_present_at=z["cities_present_at"],
            death_events=z["death_events"].reshape(-1, 2),
            capture_events=z["capture_events"].reshape(-1, 3),
            neutralize_events=z["neutralize_events"].reshape(-1, 2),
            actions_source=z["actions_source"],
            actions_dest=z["actions_dest"],
            actions_is50=z["actions_is50"],
        )
