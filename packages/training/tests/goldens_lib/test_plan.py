import numpy as np

from training.goldens.lib.content import EntryContent
from training.goldens.lib.diff import diff_content
from training.goldens.lib.entry import Changed, EntryId, Moved, New, PlannedEntry, Unchanged
from training.goldens.lib.plan import classify, match_moves


_A: EntryContent = {"x": np.arange(3)}
_B: EntryContent = {"x": np.arange(3) + 1}


def _eid(point: str, fixture: str = "f1", surface: str = "s") -> EntryId:
    return EntryId(surface=surface, point=point, fixture=fixture)


def _new(eid: EntryId, content: EntryContent) -> PlannedEntry:
    return PlannedEntry(id=eid, content=content, status=New())


def test_classify() -> None:
    assert classify(_A, None) == New()
    assert classify(_A, {"x": np.arange(3)}) == Unchanged()

    status = classify(_B, _A)
    assert isinstance(status, Changed)
    assert status.diff == diff_content(_B, _A)
    assert status.stored is _A


def test_rename_is_a_move() -> None:
    planned = [_new(_eid("p2"), _A), PlannedEntry(id=_eid("q"), content=_A, status=Unchanged())]
    out = match_moves(planned, {_eid("p1"): _A})
    assert [e.status for e in out] == [Moved(source=_eid("p1")), Unchanged()]


def test_not_a_move() -> None:
    # Equal content, but on another fixture or surface.
    orphans = {_eid("p1", fixture="f2"): _A, _eid("p1", surface="t"): _A}
    assert [e.status for e in match_moves([_new(_eid("p2"), _A)], orphans)] == [New()]

    # Same surface and fixture, different content.
    assert [e.status for e in match_moves([_new(_eid("p2"), _B)], {_eid("p1"): _A})] == [New()]


def test_each_orphan_used_once() -> None:
    planned = [_new(_eid("p2"), _A), _new(_eid("p3"), _A)]
    out = match_moves(planned, {_eid("p1"): _A})
    assert [e.status for e in out] == [Moved(source=_eid("p1")), New()]
