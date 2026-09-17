import pytest

from training.goldens.compare import compare_obs
from training.goldens.compute import compute_obs
from training.goldens.loaders import load_fixture, load_obs_digest
from training.goldens.registry import ObsEntry, obs_entries


@pytest.mark.parametrize("entry", obs_entries(), ids=lambda e: e.id)
def test_obs_golden(entry: ObsEntry) -> None:
    ref = load_obs_digest(entry.ref_path)
    if ref is None:
        pytest.fail(f"unblessed: no obs references for {entry.id}, run regen")

    game, persp = load_fixture(entry.fixture)
    got = compute_obs(game, persp, entry.point.cfg)

    mismatch = compare_obs(got, ref)
    if mismatch is not None:
        pytest.fail(mismatch.summary())
