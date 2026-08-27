import pytest

from training.bc.datapipe.walk import walk
from training.goldens.compare import compare_obs
from training.goldens.loaders import load_fixture, load_obs_reference, perspective_for
from training.goldens.registry import ObsEntry, obs_entries


@pytest.mark.parametrize("entry", obs_entries(), ids=lambda e: f"{e.point}-{e.fixture.id}")
def test_obs_golden(entry: ObsEntry) -> None:
    ref = load_obs_reference(entry.ref_path)
    if ref is None:
        pytest.fail(f"unblessed: no obs references for {entry.point}-{entry.fixture.id}, run regen")

    sim, meta = load_fixture(entry.fixture)
    persp = perspective_for(meta, entry.fixture.slot)
    frames = list(walk(sim, persp, entry.cfg))

    mismatch = compare_obs(frames, ref)
    if mismatch is not None:
        pytest.fail(mismatch.summary())
