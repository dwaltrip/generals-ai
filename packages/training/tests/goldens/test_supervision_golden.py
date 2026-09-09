import pytest

from training.bc.datapipe.sim_types import GameMeta
from training.goldens.compare import compare_supervision
from training.goldens.loaders import load_fixture, load_supervision_reference, perspective_for
from training.goldens.registry import SupervisionEntry, supervision_entries
from training.goldens.supervision_compute import compute_supervision


@pytest.mark.parametrize(
    "entry", supervision_entries(), ids=lambda e: f"{e.point}-{e.fixture.id}"
)
def test_supervision_golden(entry: SupervisionEntry) -> None:
    ref = load_supervision_reference(entry.paths)
    if ref is None:
        label = "{entry.point}-{entry.fixture.id}"
        pytest.fail(f"unblessed: no supervision references for {label}, run regen")

    sim, meta = load_fixture(entry.fixture)
    game_meta = GameMeta.from_npz(sim, meta)
    persp = perspective_for(game_meta, entry.fixture.slot)
    got = compute_supervision(sim, game_meta, persp, entry)

    mismatch = compare_supervision(got, ref, entry.keys, entry.hashed_keys)
    if mismatch is not None:
        pytest.fail(mismatch.summary())
