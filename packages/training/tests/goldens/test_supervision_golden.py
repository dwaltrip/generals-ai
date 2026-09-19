import pytest

from training.goldens.compare import compare_supervision
from training.goldens.compute import compute_supervision
from training.goldens.loaders import load_fixture, load_supervision_reference
from training.goldens.registry import SupervisionEntry, supervision_entries


@pytest.mark.parametrize("entry", supervision_entries(), ids=lambda e: e.id)
def test_supervision_golden(entry: SupervisionEntry) -> None:
    ref = load_supervision_reference(entry.refs)
    if not ref:
        pytest.fail(f"unblessed: no supervision references for {entry.id}, run regen")
    if missing := set(entry.refs) - set(ref):
        pytest.fail(
            f"missing reference files for {entry.id}: {sorted(missing)}."
            " Restore them from git, or run regen."
        )

    game, persp = load_fixture(entry.fixture)
    raw = compute_supervision(game, persp, entry.spec)

    mismatch = compare_supervision(raw, ref, entry.refs)
    if mismatch is not None:
        pytest.fail(mismatch.summary())
