import pytest

from training.goldens.fixtures import FixtureData, FixtureRecord
from training.goldens.lib import (
    Entry,
    GoldenTestContext,
    assert_entry_matches,
    entries,
    load_test_context,
    store,
)
from training.goldens.paths import REFERENCES_DIR
from training.goldens.registry import REGISTRY


type _Entry = Entry[FixtureRecord, FixtureData]
type _Context = GoldenTestContext[FixtureRecord, FixtureData]


@pytest.fixture(scope="session")
def ctx() -> _Context:
    return load_test_context(REGISTRY, REFERENCES_DIR)


def _test_id(entry: _Entry) -> str:
    return store.rel_path(entry.id).with_suffix("").as_posix()


@pytest.mark.parametrize("entry", entries(REGISTRY), ids=_test_id)
def test_golden(entry: _Entry, ctx: _Context) -> None:
    assert_entry_matches(entry, ctx)
