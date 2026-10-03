import pytest

from training.goldens.fixtures import FixtureData, FixtureRecord
from training.goldens.lib import (
    Entry,
    GoldenTestContext,
    assert_entry_matches,
    entries,
    entry_test_id,
    load_test_context,
)
from training.goldens.paths import REFERENCES_DIR
from training.goldens.registry import REGISTRY


type _Entry = Entry[FixtureRecord, FixtureData]
type _Context = GoldenTestContext[FixtureRecord, FixtureData]


@pytest.fixture(scope="session")
def ctx() -> _Context:
    return load_test_context(REGISTRY, REFERENCES_DIR)


@pytest.mark.parametrize("entry", entries(REGISTRY), ids=entry_test_id)
def test_golden(entry: _Entry, ctx: _Context) -> None:
    assert_entry_matches(entry, ctx)
