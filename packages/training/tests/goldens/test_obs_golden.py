import pytest

from training.goldens import store
from training.goldens.compare import compare_obs
from training.goldens.compute import compute_obs
from training.goldens.loaders import load_fixture
from training.goldens.registry import ObsEntry, obs_entries


@pytest.mark.parametrize("entry", obs_entries(), ids=lambda e: e.id)
def test_obs_golden(entry: ObsEntry) -> None:
    ref = store.load_obs(entry.ref)
    if ref is None:
        pytest.fail(f"unblessed: no obs references for {entry.id}, run regen")

    game, persp = load_fixture(entry.fixture)
    got = compute_obs(game, persp, entry.point.cfg)

    mismatch = compare_obs(got, ref)
    if mismatch is not None:
        pytest.fail(mismatch.summary())
