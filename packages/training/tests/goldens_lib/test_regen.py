"""
Regen on a synthetic registry: plan, apply, and re-plan around edits to a
temporary references tree, or to the registry.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest

from training.goldens.lib import store
from training.goldens.lib.apply import apply_regen
from training.goldens.lib.compute import compute_all
from training.goldens.lib.content import EntryContent, RowHashes
from training.goldens.lib.diff import (
    ArrayAdded,
    ArrayLayoutChanged,
    ArrayRemoved,
    ArrayRowsChanged,
)
from training.goldens.lib.entry import Changed, EntryId, Moved, New, Unchanged
from training.goldens.lib.hashing import row_hashes
from training.goldens.lib.plan import RegenPlan, count_statuses
from training.goldens.lib.registry import Point, Registry, Surface
from training.goldens.lib.report import render_regen_report
from training.goldens.lib.run import run_regen


# --- The synthetic registry ---


@dataclass(frozen=True)
class Fx:
    id: str
    n: int


def _load(fx: Fx) -> np.ndarray:
    return np.arange(fx.n * 3, dtype=np.int64).reshape(fx.n, 3) + fx.n


def _produce_alpha(cfg: int, data: np.ndarray) -> EntryContent:
    scaled = data * cfg
    return {"full": scaled, "hashed": row_hashes(scaled.astype(np.float32), axis=1)}


def _produce_beta(cfg: int, data: np.ndarray) -> EntryContent:
    return {"sums": data.sum(axis=1) + cfg}


_ALPHA = Surface(
    name="alpha",
    points=(Point(name="a1", cfg=1), Point(name="a2", cfg=2)),
    produce=_produce_alpha,
)
_BETA = Surface(name="beta", points=(Point(name="b1", cfg=0),), produce=_produce_beta)
_REGISTRY = Registry(
    surfaces=(_ALPHA, _BETA),
    fixtures=(Fx(id="f1", n=5), Fx(id="f2", n=7)),
    load_fixture=_load,
)

_A1_F1 = EntryId(surface="alpha", point="a1", fixture="f1")


# --- Helpers ---


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("references")
    run_regen(_REGISTRY, root, write=True)
    return root


@pytest.fixture
def tree(base: Path, tmp_path: Path) -> Path:
    root = tmp_path / "references"
    shutil.copytree(base, root)
    return root


# Also renders the report, to check that rendering doesn't fail.
def _plan(root: Path, registry: Registry = _REGISTRY) -> RegenPlan:
    plan = run_regen(registry, root, write=False)
    render_regen_report(plan)
    return plan


# Every entry not in `expected` is Unchanged.
def _assert_only(plan: RegenPlan, expected: dict[EntryId, type]) -> None:
    statuses = {e.id: type(e.status) for e in plan.planned_entries()}
    assert expected.keys() <= statuses.keys()
    assert statuses == {eid: expected.get(eid, Unchanged) for eid in statuses}


# Applies the plan and checks that the tree then re-plans as a no-op. With `base`,
# also checks that the tree is back to the base tree's bytes.
def _assert_round_trip(
    plan: RegenPlan, registry: Registry = _REGISTRY, base: Path | None = None
) -> None:
    apply_regen(plan)
    assert _plan(plan.tree.root, registry).is_noop()
    if base is not None:
        assert store.set_digest(plan.tree.root) == store.set_digest(base)


def _edit(root: Path, eid: EntryId, edit: Callable[[EntryContent], None]) -> None:
    content = store.load_references(root, {eid.surface}).get_reference(eid)
    assert content is not None
    edit(content)
    store.save(eid, content, root)


def _changed(plan: RegenPlan, eid: EntryId) -> Changed:
    (status,) = [e.status for e in plan.planned_entries() if e.id == eid]
    assert isinstance(status, Changed)
    return status


# --- Tests ---


def test_compute_all_order() -> None:
    seen = []
    computed = compute_all(_REGISTRY, on_fixture=lambda fx, data: seen.append(fx.id))
    assert seen == ["f1", "f2"]
    assert list(computed) == ["alpha", "beta"]
    assert [c.id for c in computed["alpha"]] == [
        EntryId("alpha", "a1", "f1"),
        EntryId("alpha", "a1", "f2"),
        EntryId("alpha", "a2", "f1"),
        EntryId("alpha", "a2", "f2"),
    ]


def test_base_tree_is_clean(base: Path) -> None:
    plan = _plan(base)
    assert plan.is_noop()
    assert [s.surface for s in plan.surfaces] == ["alpha", "beta"]
    assert [f.id for f in plan.fixtures] == ["f1", "f2"]
    assert not plan.tree.unrecognized and not plan.tree.ignored


def test_deleted_file_is_new(tree: Path, base: Path) -> None:
    store.remove(_A1_F1, tree)

    plan = _plan(tree)
    _assert_only(plan, {_A1_F1: New})
    assert count_statuses(plan.planned_entries())[New] == 1
    _assert_round_trip(plan, base=base)


def test_renamed_point_is_a_move(tree: Path, base: Path) -> None:
    (tree / "alpha" / "a1").rename(tree / "alpha" / "old")

    plan = _plan(tree)
    expected = {
        EntryId("alpha", "a1", f): Moved(source=EntryId("alpha", "old", f)) for f in ("f1", "f2")
    }
    _assert_only(plan, {eid: Moved for eid in expected})
    assert {e.id: e.status for e in plan.planned_entries() if e.id in expected} == expected
    assert all(not s.planned_removals for s in plan.surfaces)
    _assert_round_trip(plan, base=base)


def test_edited_rows_are_changed(tree: Path, base: Path) -> None:
    def edit(content: EntryContent) -> None:
        full, hashed = content["full"], content["hashed"]
        assert isinstance(full, np.ndarray) and isinstance(hashed, RowHashes)
        full[[1, 3]] += 1
        hashed.hashes[0] ^= np.uint64(1)

    _edit(tree, _A1_F1, edit)

    plan = _plan(tree)
    _assert_only(plan, {_A1_F1: Changed})
    assert plan.changed_surfaces() == {"alpha"}
    changes = _changed(plan, _A1_F1).diff.changes
    rows = {c.name: c.rows for c in changes if isinstance(c, ArrayRowsChanged)}
    assert len(changes) == 2 and rows == {"full": (1, 3), "hashed": (0,)}
    _assert_round_trip(plan, base=base)


def test_extra_and_missing_arrays(tree: Path, base: Path) -> None:
    def edit(content: EntryContent) -> None:
        del content["full"]
        content["extra"] = np.zeros(2)

    _edit(tree, _A1_F1, edit)

    plan = _plan(tree)
    _assert_only(plan, {_A1_F1: Changed})
    changes = _changed(plan, _A1_F1).diff.changes
    assert [(type(c), c.name) for c in changes] == [(ArrayRemoved, "extra"), (ArrayAdded, "full")]
    _assert_round_trip(plan, base=base)


def test_edited_meta_layout(tree: Path, base: Path) -> None:
    def edit(content: EntryContent) -> None:
        hashed = content["hashed"]
        assert isinstance(hashed, RowHashes)
        layout = replace(hashed.layout, dtype=np.dtype(np.float16))
        content["hashed"] = RowHashes(hashes=hashed.hashes, layout=layout)

    _edit(tree, _A1_F1, edit)

    plan = _plan(tree)
    _assert_only(plan, {_A1_F1: Changed})
    changes = _changed(plan, _A1_F1).diff.changes
    assert [(type(c), c.name) for c in changes] == [(ArrayLayoutChanged, "hashed")]
    _assert_round_trip(plan, base=base)


def test_case_only_rename_rejected(tree: Path) -> None:
    (tree / "alpha" / "a1").rename(tree / "alpha" / "A1")
    with pytest.raises(ValueError, match="only by case"):
        _plan(tree)


def test_invalid_reference_rejected(tmp_path: Path) -> None:
    run_regen(_REGISTRY, tmp_path, write=True)
    (tmp_path / store.rel_path(_A1_F1)).write_bytes(b"")
    with pytest.raises(ValueError, match="invalid references"):
        _plan(tmp_path)


def test_no_fixtures_rejected(tree: Path) -> None:
    registry = replace(_REGISTRY, fixtures=())
    with pytest.raises(ValueError, match="no fixtures"):
        _plan(tree, registry)


def test_dropped_point_is_removed(tree: Path) -> None:
    registry = replace(_REGISTRY, surfaces=(replace(_ALPHA, points=_ALPHA.points[1:]), _BETA))

    plan = _plan(tree, registry)
    _assert_only(plan, {})
    alpha, beta = plan.surfaces
    assert alpha.planned_removals == [EntryId("alpha", "a1", "f1"), EntryId("alpha", "a1", "f2")]
    assert beta.planned_removals == []
    _assert_round_trip(plan, registry)
    assert not (tree / "alpha" / "a1" / "f1.npz").exists()


def test_surface_with_no_points(tree: Path) -> None:
    registry = replace(_REGISTRY, surfaces=(_ALPHA, replace(_BETA, points=())))

    plan = _plan(tree, registry)
    beta = plan.surfaces[1]
    assert beta.surface == "beta" and beta.planned_entries == []
    assert beta.planned_removals == [EntryId("beta", "b1", "f1"), EntryId("beta", "b1", "f2")]
    _assert_round_trip(plan, registry)


def test_stray_files(tree: Path) -> None:
    strays = ["alpha/notes.md", "alpha/a1/f1.npy", "alpha/f1.npz", "gamma/g1/f1.npz"]
    for rel in strays:
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_bytes(b"x")

    plan = _plan(tree)
    assert plan.is_noop()
    assert plan.tree.unrecognized == [Path("alpha/f1.npz"), Path("gamma/g1/f1.npz")]
    assert plan.tree.ignored == [Path("alpha/a1/f1.npy"), Path("alpha/notes.md")]
    _assert_round_trip(plan)
    assert all((tree / rel).exists() for rel in strays)


def test_unchanged_file_not_rewritten(tree: Path, base: Path) -> None:
    # Rewrite one file with equal content in a different byte form: members out of
    # order, and a Fortran-ordered array.
    path = tree / store.rel_path(_A1_F1)
    with np.load(path) as z:
        members: dict[str, Any] = {name: z[name] for name in z.files}
    members["full"] = np.asfortranarray(members["full"])
    np.savez(path, **dict(reversed(members.items())))

    before = path.read_bytes()
    assert before != (base / store.rel_path(_A1_F1)).read_bytes()

    plan = run_regen(_REGISTRY, tree, write=True)
    assert plan.is_noop()
    assert path.read_bytes() == before
