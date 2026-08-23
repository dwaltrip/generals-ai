import pytest

from training.goldens.compare import compare_targets
from training.goldens.loaders import load_fixture, load_targets_bundle, perspective_for
from training.goldens.registry import TargetsEntry, targets_entries
from training.goldens.targets_compute import compute_targets


@pytest.mark.parametrize("entry", targets_entries(), ids=lambda e: f"{e.point}-{e.fixture.id}")
def test_targets_golden(entry: TargetsEntry) -> None:
    bundle = load_targets_bundle(entry.bundle_path)
    if bundle is None:
        pytest.fail(f"unblessed: no targets bundle for {entry.fixture.id}, run regen")

    sim, meta = load_fixture(entry.fixture)
    persp = perspective_for(meta, entry.fixture.slot)
    got = compute_targets(sim, persp, entry)

    mismatch = compare_targets(got, bundle, entry.keys)
    if mismatch is not None:
        pytest.fail(mismatch.summary())
