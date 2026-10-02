from __future__ import annotations

from collections.abc import Callable
from typing import Any

from training.goldens.lib.content import EntryContent, check_content
from training.goldens.lib.entry import ComputedEntriesBySurface, ComputedEntry, EntryId
from training.goldens.lib.registry import Entry, Fixture, Registry, entries


def produce[DataT](entry: Entry[Any, DataT], data: DataT) -> EntryContent:
    content = entry.surface.produce(entry.point.cfg, data)
    check_content(content)
    return content


# Returns the computed entries grouped by surface, in registry order.
# Surfaces without points are included (planning needs the list of registered surfaces)
def compute_all[FixtureT: Fixture, DataT](
    registry: Registry[FixtureT, DataT],
    *,
    on_fixture: Callable[[FixtureT, DataT], None] | None = None,
) -> ComputedEntriesBySurface:
    all_entries = entries(registry)
    contents: dict[EntryId, EntryContent] = {}
    for fixture in registry.fixtures:
        data = registry.load_fixture(fixture)
        for entry in all_entries:
            if entry.fixture.id == fixture.id:
                contents[entry.id] = produce(entry, data)
        if on_fixture is not None:
            on_fixture(fixture, data)

    computed: ComputedEntriesBySurface = {s.name: [] for s in registry.surfaces}
    for entry in all_entries:
        computed[entry.surface.name].append(ComputedEntry(id=entry.id, content=contents[entry.id]))
    return computed
