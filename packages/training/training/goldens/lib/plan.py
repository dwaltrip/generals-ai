from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from training.goldens.lib import store
from training.goldens.lib.content import EntryContent
from training.goldens.lib.diff import diff_content
from training.goldens.lib.entry import (
    Changed,
    ComputedEntriesBySurface,
    ComputedEntry,
    EntryId,
    FixtureInfo,
    Moved,
    New,
    PlannedEntry,
    PlannedEntryStatus,
    Unchanged,
)
from training.goldens.lib.store import ReferenceTree


@dataclass(frozen=True)
class SurfacePlan:
    surface: str
    planned_entries: list[PlannedEntry]
    planned_removals: list[EntryId]


@dataclass(frozen=True)
class RegenPlan:
    tree: ReferenceTree             # the references dir, before the run
    fixtures: list[FixtureInfo]     # registry order
    surfaces: list[SurfacePlan]     # registry order (includes surfaces without points)

    def planned_entries(self) -> list[PlannedEntry]:
        return [e for s in self.surfaces for e in s.planned_entries]

    # Every status is Unchanged and nothing is removed.
    def is_noop(self) -> bool:
        all_unchanged = all(isinstance(e.status, Unchanged) for e in self.planned_entries())
        return all_unchanged and not any(s.planned_removals for s in self.surfaces)

    # Surfaces with any Changed status.
    def changed_surfaces(self) -> set[str]:
        return {
            s.surface
            for s in self.surfaces
            if any(isinstance(e.status, Changed) for e in s.planned_entries)
        }


def count_statuses(
    planned_entries: Iterable[PlannedEntry],
) -> Counter[type[PlannedEntryStatus]]:
    return Counter(type(e.status) for e in planned_entries)


def plan_regen(
    computed: ComputedEntriesBySurface,
    fixtures: list[FixtureInfo],
    root: Path,
) -> RegenPlan:
    # TODO: possibly a custom "regen" error type later on.
    if not fixtures:
        raise ValueError("no fixtures, so every reference would be planned for removal")

    tree = store.load_references(root, computed.keys())
    if tree.invalid_refs:
        lines = [f"  {store.rel_path(eid)}: {err}" for eid, err in tree.invalid_refs.items()]
        raise ValueError(
            "invalid references. Restore them from git, or delete them and re-run:\n"
            + "\n".join(lines)
        )
    orphans = _find_orphans_by_surface(tree, computed)

    surfaces = [
        _plan_surface(name, entries, tree, orphans[name])
        for name, entries in computed.items()
    ]
    return RegenPlan(tree=tree, fixtures=fixtures, surfaces=surfaces)


def _plan_surface(
    name: str,
    computed: list[ComputedEntry],
    tree: ReferenceTree,
    orphans: dict[EntryId, EntryContent],
) -> SurfacePlan:
    assert all(c.id.surface == name for c in computed)
    assert all(eid.surface == name for eid in orphans)

    planned = [
        PlannedEntry(
            id=c.id, content=c.content, status=classify(c.content, tree.get_reference(c.id))
        )
        for c in computed
    ]
    planned = match_moves(planned, orphans)

    sources = {e.status.source for e in planned if isinstance(e.status, Moved)}
    return SurfacePlan(
        surface=name,
        planned_entries=planned,
        planned_removals=[eid for eid in orphans if eid not in sources],
    )


# References on disk that no computed entry claims, grouped by surface. Every
# computed surface has a key. An orphan that is the same file as a computed entry
# under a different spelling can't be planned around.
def _find_orphans_by_surface(
    tree: ReferenceTree, computed: ComputedEntriesBySurface
) -> dict[str, dict[EntryId, EntryContent]]:
    computed_ids = {c.id for cs in computed.values() for c in cs}
    orphans = {eid: ref for eid, ref in tree.references.items() if eid not in computed_ids}

    claimed = {store.path_key(eid): eid for eid in computed_ids}
    clashes = [(claimed[k], eid) for eid in orphans if (k := store.path_key(eid)) in claimed]
    if clashes:
        lines = [
            f"  {store.rel_path(o)} on disk, {store.rel_path(c)} registered" for c, o in clashes
        ]
        raise ValueError(
            "references differ from registered entries only by case."
            " Rename them with `git mv` first:\n" + "\n".join(lines)
        )

    by_surface: dict[str, dict[EntryId, EntryContent]] = {name: {} for name in computed}
    for eid, ref in orphans.items():
        by_surface[eid.surface][eid] = ref
    return by_surface


def classify(got: EntryContent, stored: EntryContent | None) -> New | Unchanged | Changed:
    if stored is None:
        return New()
    diff = diff_content(got, stored)
    return Unchanged() if diff is None else Changed(diff=diff, stored=stored)


# "New" entries become "Moved" when there is a matching orphan (same surface,
# same fixture, equal content). Each orphan is used at most once, in the order given.
def match_moves(
    planned_entries: list[PlannedEntry],
    orphans: dict[EntryId, EntryContent],
) -> list[PlannedEntry]:
    unused = dict(orphans)
    out = []
    for entry in planned_entries:
        if isinstance(entry.status, New):
            source = _find_move_source(entry, unused)
            if source is not None:
                del unused[source]
                entry = replace(entry, status=Moved(source=source))
        out.append(entry)
    return out


def _find_move_source(entry: PlannedEntry, orphans: dict[EntryId, EntryContent]) -> EntryId | None:
    for eid, content in orphans.items():
        same_slot = eid.surface == entry.id.surface and eid.fixture == entry.id.fixture
        if same_slot and diff_content(entry.content, content) is None:
            return eid
    return None
