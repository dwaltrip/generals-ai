from __future__ import annotations

from dataclasses import dataclass

from training.goldens.lib.content import EntryContent
from training.goldens.lib.diff import ContentDiff


@dataclass(frozen=True)
class EntryId:
    surface: str
    point: str
    fixture: str


@dataclass(frozen=True)
class FixtureInfo:
    id: str


@dataclass(frozen=True)
class ComputedEntry:
    id: EntryId
    content: EntryContent


type ComputedEntriesBySurface = dict[str, list[ComputedEntry]]


# --- Planning statuses ---


@dataclass(frozen=True)
class New:
    pass


@dataclass(frozen=True)
class Unchanged:
    pass


@dataclass(frozen=True)
class Changed:
    diff: ContentDiff
    stored: EntryContent


@dataclass(frozen=True)
class Moved:
    source: EntryId


type PlannedEntryStatus = New | Unchanged | Changed | Moved


@dataclass(frozen=True)
class PlannedEntry:
    id: EntryId
    content: EntryContent
    status: PlannedEntryStatus
