"""Helpers for the project's golden tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.goldens.lib import store
from training.goldens.lib.compute import produce
from training.goldens.lib.entry import Changed, New, Unchanged
from training.goldens.lib.plan import classify
from training.goldens.lib.registry import Entry, Fixture, Registry
from training.goldens.lib.render import render_content_diff
from training.goldens.lib.store import ReferenceTree


# Built once per test session and passed to each golden test.
@dataclass(frozen=True, kw_only=True)
class GoldenTestContext[FixtureT: Fixture, DataT]:
    tree: ReferenceTree
    load_fixture: Callable[[FixtureT], DataT]   # loads each fixture once


def load_test_context[FixtureT: Fixture, DataT](
    registry: Registry[FixtureT, DataT], root: Path
) -> GoldenTestContext[FixtureT, DataT]:
    loaded: dict[str, DataT] = {}

    def load_fixture(fixture: FixtureT) -> DataT:
        if fixture.id not in loaded:
            loaded[fixture.id] = registry.load_fixture(fixture)
        return loaded[fixture.id]

    return GoldenTestContext(
        tree=store.load_references(root, [s.name for s in registry.surfaces]),
        load_fixture=load_fixture,
    )


# An id for an entry's test case, such as `obs/fp16_player_status_on/ukRz7oSS8-s7`.
def entry_test_id(entry: Entry[Any, Any]) -> str:
    return store.rel_path(entry.id).with_suffix("").as_posix()


def assert_entry_matches[FixtureT: Fixture, DataT](
    entry: Entry[FixtureT, DataT], ctx: GoldenTestContext[FixtureT, DataT]
) -> None:
    if (err := ctx.tree.invalid_refs.get(entry.id)) is not None:
        raise AssertionError(
            f"invalid reference {store.rel_path(entry.id)}: {err}."
            " Restore it from git, or delete it and run regen."
        )

    got = produce(entry, ctx.load_fixture(entry.fixture))
    match classify(got, ctx.tree.get_reference(entry.id)):
        case New():
            raise AssertionError(
                f"unblessed: no reference for {store.rel_path(entry.id)}, run regen"
            )
        case Changed(diff=diff):
            raise AssertionError(render_content_diff(diff))
        case Unchanged():
            pass
