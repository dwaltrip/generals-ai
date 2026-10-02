from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from training.goldens.lib.content import EntryContent
from training.goldens.lib.entry import EntryId, FixtureInfo


@dataclass(frozen=True, kw_only=True)
class Point[CfgT]:
    name: str
    cfg: CfgT


@dataclass(frozen=True, kw_only=True)
class Surface[CfgT, DataT]:
    name: str
    points: tuple[Point[CfgT], ...]
    produce: Callable[[CfgT, DataT], EntryContent]


class Fixture(Protocol):
    @property
    def id(self) -> str: ...


@dataclass(frozen=True, kw_only=True)
class Registry[FixtureT: Fixture, DataT]:
    surfaces: tuple[Surface[Any, DataT], ...]
    fixtures: tuple[FixtureT, ...]
    load_fixture: Callable[[FixtureT], DataT]

    def __post_init__(self) -> None:
        _check_path_names("surface", [s.name for s in self.surfaces])
        for s in self.surfaces:
            _check_path_names(f"{s.name} point", [p.name for p in s.points])
        _check_path_names("fixture", [f.id for f in self.fixtures])


# Names become path components. Duplicates are found with casefold(), since the
# default macOS file system ignores case.
def _check_path_names(kind: str, names: list[str]) -> None:
    for name in names:
        if not name or "/" in name or name.startswith("."):
            raise ValueError(f"bad {kind} name: {name!r}")

    counts = Counter(n.casefold() for n in names)
    dupes = [n for n in names if counts[n.casefold()] > 1]
    if dupes:
        raise ValueError(f"duplicate {kind} names (ignoring case): {dupes}")


@dataclass(frozen=True)
class Entry[FixtureT: Fixture, DataT]:
    surface: Surface[Any, DataT]
    point: Point[Any]
    fixture: FixtureT

    @property
    def id(self) -> EntryId:
        return EntryId(surface=self.surface.name, point=self.point.name, fixture=self.fixture.id)


def entries[FixtureT: Fixture, DataT](
    registry: Registry[FixtureT, DataT],
) -> list[Entry[FixtureT, DataT]]:
    return [
        Entry(surface=s, point=p, fixture=f)
        for s in registry.surfaces
        for p in s.points
        for f in registry.fixtures
    ]


def fixture_infos(registry: Registry) -> list[FixtureInfo]:
    return [FixtureInfo(id=f.id) for f in registry.fixtures]
