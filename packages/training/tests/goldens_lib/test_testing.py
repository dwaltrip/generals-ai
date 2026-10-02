from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from training.goldens.lib import store
from training.goldens.lib.content import EntryContent
from training.goldens.lib.registry import Point, Registry, Surface, entries
from training.goldens.lib.run import run_regen
from training.goldens.lib.testing import assert_entry_matches, load_test_context


@dataclass(frozen=True)
class Fx:
    id: str


def _produce(cfg: int, data: np.ndarray) -> EntryContent:
    return {"scaled": data * cfg}


# Records each fixture load in `loads`.
def _registry(loads: list[str]) -> Registry[Fx, np.ndarray]:
    def load(fx: Fx) -> np.ndarray:
        loads.append(fx.id)
        return np.arange(4)

    points = (Point(name="p", cfg=2), Point(name="q", cfg=3))
    return Registry(
        surfaces=(Surface(name="s", points=points, produce=_produce),),
        fixtures=(Fx(id="f"),),
        load_fixture=load,
    )


def test_assert_entry_matches(tmp_path: Path) -> None:
    loads: list[str] = []
    registry = _registry(loads)
    p, q = entries(registry)
    run_regen(registry, tmp_path, write=True)

    # Both entries pass, and share one load of their fixture.
    loads.clear()
    ctx = load_test_context(registry, tmp_path)
    assert_entry_matches(p, ctx)
    assert_entry_matches(q, ctx)
    assert loads == ["f"]

    store.save(p.id, {"scaled": np.array([0, 2, 4, 7])}, tmp_path)
    with pytest.raises(AssertionError, match=r"scaled: row 3 of 4"):
        assert_entry_matches(p, load_test_context(registry, tmp_path))

    (tmp_path / store.rel_path(p.id)).write_bytes(b"")
    ctx = load_test_context(registry, tmp_path)
    with pytest.raises(AssertionError, match="invalid reference"):
        assert_entry_matches(p, ctx)
    assert_entry_matches(q, ctx)

    store.remove(p.id, tmp_path)
    with pytest.raises(AssertionError, match="unblessed"):
        assert_entry_matches(p, load_test_context(registry, tmp_path))
