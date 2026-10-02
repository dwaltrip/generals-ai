from __future__ import annotations

from dataclasses import dataclass

from training.bc.datapipe.sim_types import CorpusGame, PerspectiveMeta, SimGame
from training.goldens.paths import FIXTURES_DIR


@dataclass(frozen=True, kw_only=True)
class FixtureRecord:
    replay_id: str
    perspective_slot: int
    note: str

    @property
    def id(self) -> str:
        return f"{self.replay_id}-s{self.perspective_slot}"


@dataclass(frozen=True)
class FixtureData:
    game: SimGame
    persp: PerspectiveMeta


def load_fixture(fixture: FixtureRecord) -> FixtureData:
    corpus = CorpusGame.load(FIXTURES_DIR / f"{fixture.replay_id}.npz")
    return FixtureData(
        game=corpus.sim,
        persp=corpus.perspective_for_slot(fixture.perspective_slot),
    )
