import pytest

from training.bc.datapipe.sim_types import GameMeta
from training.goldens.compare import compare_supervision
from training.goldens.loaders import load_fixture, load_supervision_bundle, perspective_for
from training.goldens.registry import SupervisionEntry, supervision_entries
from training.goldens.supervision_compute import compute_supervision


@pytest.mark.parametrize(
    "entry", supervision_entries(), ids=lambda e: f"{e.point}-{e.fixture.id}"
)
def test_supervision_golden(entry: SupervisionEntry) -> None:
    bundle = load_supervision_bundle(entry.bundle_path)
    if bundle is None:
        pytest.fail(f"unblessed: no supervision bundle for {entry.fixture.id}, run regen")

    sim, meta = load_fixture(entry.fixture)
    game_meta = GameMeta.from_npz(sim, meta)
    persp = perspective_for(meta, entry.fixture.slot)
    got = compute_supervision(sim, game_meta, persp, entry)

    mismatch = compare_supervision(got, bundle, entry.keys)
    if mismatch is not None:
        pytest.fail(mismatch.summary())
