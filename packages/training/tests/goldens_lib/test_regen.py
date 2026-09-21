"""
Regen against a temporary references tree: plan, apply, and re-plan around a
few edits to the tree. The base tree is blessed from current compute, so these
tests don't read the committed references and stay green when the goldens fire.
"""

from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest

from training.goldens import store
from training.goldens.regen.apply import apply
from training.goldens.regen.plan import Plan, Status, assemble, plan_fixture
from training.goldens.regen.report import render_report
from training.goldens.registry import FIXTURES, obs_entries, supervision_entries
from training.goldens.store import RefId


def _plan(root: Path) -> Plan:
    return assemble([plan_fixture(fx, root) for fx in FIXTURES], root)


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("references")
    apply(_plan(root), root)
    return root


@pytest.fixture
def tree(base: Path, tmp_path: Path) -> Path:
    root = tmp_path / "references"
    shutil.copytree(base, root)
    return root


# One obs reference, and one supervision key's reference on every fixture.
_OBS_RID = obs_entries()[0].ref
_SUP_RID = next(iter(supervision_entries()[0].refs.values())).ref
_SUP_RIDS = [replace(_SUP_RID, fixture=fx.id) for fx in FIXTURES]


def _statuses(plan: Plan) -> dict[RefId, Status]:
    return {i.ref: i.status for i in [*plan.obs, *plan.supervision]}


def _assert_only(plan: Plan, expected: dict[RefId, Status]) -> None:
    statuses = _statuses(plan)
    assert expected.keys() <= statuses.keys()
    for rid, status in statuses.items():
        assert status is expected.get(rid, Status.UNCHANGED), rid


def _assert_round_trip(plan: Plan, root: Path, base: Path) -> None:
    render_report(plan, root)
    apply(plan, root)
    assert _plan(root).is_noop()
    assert store.set_digest(root) == store.set_digest(base)


# --- Tests ---


def test_base_tree_is_clean(base: Path) -> None:
    plan = _plan(base)
    assert plan.is_noop()
    assert not plan.unrecognized and not plan.ignored


def test_deleted_files_are_new(tree: Path, base: Path) -> None:
    store.remove(_OBS_RID, tree)
    store.remove(_SUP_RID, tree)

    plan = _plan(tree)
    _assert_only(plan, {_OBS_RID: Status.NEW, _SUP_RID: Status.NEW})
    assert plan.removed == [] and plan.byte_matches == []
    _assert_round_trip(plan, tree, base)


def test_representative_rename_is_a_move(tree: Path, base: Path) -> None:
    old = [replace(rid, point="old_rep") for rid in _SUP_RIDS]
    for rid, old_rid in zip(_SUP_RIDS, old, strict=True):
        store.ref_path(rid, tree).rename(store.ref_path(old_rid, tree))

    plan = _plan(tree)
    _assert_only(plan, {rid: Status.MOVED for rid in _SUP_RIDS})
    moved_from = {s.ref: s.moved_from for s in plan.supervision if s.status is Status.MOVED}
    assert moved_from == dict(zip(_SUP_RIDS, old, strict=True))
    assert plan.removed == [] and plan.byte_matches == []
    _assert_round_trip(plan, tree, base)


def test_key_rename_is_new_plus_removed(tree: Path, base: Path) -> None:
    old = [replace(rid, key="old_key") for rid in _SUP_RIDS]
    for rid, old_rid in zip(_SUP_RIDS, old, strict=True):
        store.ref_path(rid, tree).rename(store.ref_path(old_rid, tree))

    plan = _plan(tree)
    _assert_only(plan, {rid: Status.NEW for rid in _SUP_RIDS})
    assert set(plan.removed) == set(old)
    assert set(plan.byte_matches) == set(zip(_SUP_RIDS, old, strict=True))
    _assert_round_trip(plan, tree, base)


def test_edited_files_are_changed(tree: Path, base: Path) -> None:
    rows = [3, 10, 11, 12]
    frames = [0, 1, 2, 3, 4]
    channels = [2, 5]

    path = store.ref_path(_SUP_RID, tree)
    arr = np.load(path)
    raw = arr.view(np.uint8).reshape(arr.shape[0], -1)
    raw[rows, 0] ^= 0xFF
    np.save(path, arr)

    path = store.ref_path(_OBS_RID, tree)
    with np.load(path) as z:
        fh, ch = z["frame_hashes"].copy(), z["channel_hashes"].copy()
    fh[frames] ^= 1
    ch[channels] ^= 1
    np.savez(path, frame_hashes=fh, channel_hashes=ch)

    plan = _plan(tree)
    _assert_only(plan, {_OBS_RID: Status.CHANGED, _SUP_RID: Status.CHANGED})
    (sup,) = [s for s in plan.supervision if s.status is Status.CHANGED]
    assert sup.diff is not None and sup.diff.changed_rows.tolist() == rows
    (obs,) = [o for o in plan.obs if o.status is Status.CHANGED]
    assert obs.diff is not None
    assert obs.diff.changed_frames is not None and obs.diff.changed_frames.tolist() == frames
    assert obs.diff.changed_channels is not None and obs.diff.changed_channels.tolist() == channels
    _assert_round_trip(plan, tree, base)
