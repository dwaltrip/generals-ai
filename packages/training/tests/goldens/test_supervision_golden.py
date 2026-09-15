import pytest

from training.goldens.compare import compare_supervision, to_stored
from training.goldens.compute import compute_supervision
from training.goldens.loaders import load_fixture, load_supervision_reference
from training.goldens.registry import SupervisionEntry, supervision_entries


@pytest.mark.parametrize(
    "entry", supervision_entries(), ids=lambda e: f"{e.point}-{e.fixture.id}"
)
def test_supervision_golden(entry: SupervisionEntry) -> None:
    ref = load_supervision_reference(entry.paths)
    if ref is None:
        label = f"{entry.point}-{entry.fixture.id}"
        pytest.fail(f"unblessed: no supervision references for {label}, run regen")

    game, persp = load_fixture(entry.fixture)
    got = to_stored(compute_supervision(game, persp, entry.spec), entry.hashed_keys)

    mismatch = compare_supervision(got, ref, entry.keys)
    if mismatch is not None:
        pytest.fail(mismatch.summary())
