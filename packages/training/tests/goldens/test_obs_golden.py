import pytest

from training.bc.datapipe.sim_types import GameMeta
from training.goldens.compare import compare_obs
from training.goldens.compute import compute_obs
from training.goldens.loaders import load_fixture, load_obs_digest, perspective_for
from training.goldens.registry import ObsEntry, obs_entries


@pytest.mark.parametrize("entry", obs_entries(), ids=lambda e: f"{e.point}-{e.fixture.id}")
def test_obs_golden(entry: ObsEntry) -> None:
    ref = load_obs_digest(entry.ref_path)
    if ref is None:
        pytest.fail(f"unblessed: no obs references for {entry.point}-{entry.fixture.id}, run regen")

    sim, meta = load_fixture(entry.fixture)
    game_meta = GameMeta.from_npz(sim, meta)
    persp = perspective_for(game_meta, entry.fixture.slot)
    got = compute_obs(sim, game_meta, persp, entry.cfg)

    mismatch = compare_obs(got, ref)
    if mismatch is not None:
        pytest.fail(mismatch.summary())
