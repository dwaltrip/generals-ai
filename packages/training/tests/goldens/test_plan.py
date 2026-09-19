"""
Move detection on a temporary references tree: a strict move (same fixture and
key, different point), a byte coincidence across keys that must not become a
move, and a plain removal.
"""

from pathlib import Path

import numpy as np

from training.goldens import store
from training.goldens.regen.plan import PlannedSupervision, Status, _detect_moves
from training.goldens.registry import FixtureRecord, RefForm, SupervisionKey
from training.goldens.store import RefId, Surface


def _key(name: str) -> SupervisionKey:
    return SupervisionKey(name=name, deps=(), form=RefForm.FULL)


def _rid(point: str, key: str, fixture: str = "fx-s1") -> RefId:
    return RefId(Surface.SUPERVISION, point=point, fixture=fixture, key=key)


def _new(point: str, key: str, array: np.ndarray) -> PlannedSupervision:
    return PlannedSupervision(
        ref=_rid(point, key),
        key=_key(key),
        fixture=FixtureRecord(replay_id="fx", slot=1, note=""),
        gid=(),
        points=(point,),
        array=array,
        status=Status.NEW,
    )


def test_detect_moves(tmp_path: Path) -> None:
    a = np.arange(6, dtype=np.int64)
    b = np.arange(6, dtype=np.int64) + 100

    # Orphans: the true predecessor of alive (same key), a different key with the
    # same bytes, and one with nothing to match.
    store.save_supervision(_rid("old", "alive"), a, tmp_path)
    store.save_supervision(_rid("other", "present"), a, tmp_path)
    store.save_supervision(_rid("old", "gone"), b, tmp_path)
    orphans = [_rid("other", "present"), _rid("old", "alive"), _rid("old", "gone")]

    planned = [_new("renamed", "alive", a), _new("renamed", "fresh", a)]
    obs, sup, removed, byte_matches = _detect_moves([], planned, orphans, tmp_path)

    assert obs == []
    alive, fresh = sup
    assert alive.status is Status.MOVED
    assert alive.moved_from == _rid("old", "alive")
    assert fresh.status is Status.NEW
    assert fresh.moved_from is None

    assert removed == [_rid("other", "present"), _rid("old", "gone")]
    assert byte_matches == [(_rid("renamed", "fresh"), _rid("other", "present"))]
