from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from training.bc.datapipe.sim_types import CorpusGame, PerspectiveMeta, SimGame
from training.goldens import store
from training.goldens.paths import FIXTURES_DIR
from training.goldens.registry import FixtureRecord, KeyRef


def load_fixture(fixture: FixtureRecord) -> tuple[SimGame, PerspectiveMeta]:
    corpus = CorpusGame.load(FIXTURES_DIR / f"{fixture.replay_id}.npz")
    return corpus.sim, corpus.perspective_for_slot(fixture.slot)


# Returns the arrays found on disk, which may be a subset of `refs`.
def load_supervision_reference(refs: Mapping[str, KeyRef]) -> dict[str, np.ndarray]:
    out = {}
    for ref in refs.values():
        arr = store.load_supervision(ref.ref)
        if arr is not None:
            out[ref.key] = arr
    return out
