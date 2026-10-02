from dataclasses import dataclass

import numpy as np
import pytest

from training.goldens.lib.content import EntryContent
from training.goldens.lib.entry import EntryId
from training.goldens.lib.registry import Point, Registry, Surface, entries


@dataclass(frozen=True)
class Fx:
    id: str


def _produce(cfg: int, data: int) -> EntryContent:
    return {"x": np.array([cfg + data])}


def _registry(surfaces: dict[str, list[str]], fixtures: list[str]) -> Registry[Fx, int]:
    return Registry(
        surfaces=tuple(
            Surface(
                name=s,
                points=tuple(Point(name=p, cfg=i) for i, p in enumerate(points)),
                produce=_produce,
            )
            for s, points in surfaces.items()
        ),
        fixtures=tuple(Fx(id=f) for f in fixtures),
        load_fixture=lambda fx: len(fx.id),
    )


def test_entries_order() -> None:
    reg = _registry({"s1": ["p1", "p2"], "s2": ["p1"]}, ["f1", "f2"])
    assert [e.id for e in entries(reg)] == [
        EntryId("s1", "p1", "f1"),
        EntryId("s1", "p1", "f2"),
        EntryId("s1", "p2", "f1"),
        EntryId("s1", "p2", "f2"),
        EntryId("s2", "p1", "f1"),
        EntryId("s2", "p1", "f2"),
    ]


@pytest.mark.parametrize(
    ("surfaces", "fixtures"),
    [
        ({"s": ["p", "P"]}, ["f"]),   # duplicates ignoring case
        ({"s": ["p"]}, ["f", "f"]),
        ({"s": ["a/b"]}, ["f"]),
        ({"s": [""]}, ["f"]),
        ({".s": ["p"]}, ["f"]),
    ],
)
def test_bad_names_rejected(surfaces: dict[str, list[str]], fixtures: list[str]) -> None:
    with pytest.raises(ValueError):
        _registry(surfaces, fixtures)
