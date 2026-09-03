#!/usr/bin/env -S uv run python
"""Copy a few random eligible corpus games into the goldens fixtures dir and
print FixtureRecord lines to paste into the registry. Throwaway picks for the
build; proper fixture selection replaces them. Run from packages/training:

    ./scripts/goldens_pick_throwaway_fixtures.py [SEED] [N]
"""

from __future__ import annotations

from pathlib import Path
import random
import shutil
import sys

import numpy as np

from settings import INTERMEDIATE_DIR
from training.bc.filters import eligible_perspectives
from training.bc.splits import load_curated_names
from training.bc.utils import list_sim_paths, meta_path_for
from training.goldens.registry import FIXTURES_DIR


def main(seed: int, n: int) -> None:
    rng = random.Random(seed)
    curated = load_curated_names()
    paths = list_sim_paths(INTERMEDIATE_DIR)
    rng.shuffle(paths)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    picked = 0
    for sim_path in paths:
        meta_path = meta_path_for(sim_path)
        ks = eligible_perspectives(sim_path, meta_path, curated)
        if not ks:
            continue
        k = rng.choice(ks)
        with np.load(meta_path) as meta:
            slot = int(meta["perspective_player_ids"][k])
            elim_t = int(meta["elim_timestep"][k])
        shutil.copy(sim_path, FIXTURES_DIR / sim_path.name)
        shutil.copy(meta_path, FIXTURES_DIR / meta_path.name)
        note = "throwaway pick" + (f", eliminated at t={elim_t}" if elim_t != -1 else ", survivor")
        print(f'    FixtureRecord(replay_id="{sim_path.stem}", slot={slot}, note="{note}"),')
        picked += 1
        if picked >= n:
            break


if __name__ == "__main__":
    main(
        int(sys.argv[1]) if len(sys.argv) > 1 else 0,
        int(sys.argv[2]) if len(sys.argv) > 2 else 2,
    )
