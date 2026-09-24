"""Plan step for regen: compute and classify references, detect changes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations
from pathlib import Path

import numpy as np

from training.bc.datapipe.sim_types import PerspectiveMeta, SimGame
from training.goldens import store
from training.goldens.compare import (
    KeyDiff,
    ObsMismatch,
    compare_obs,
    diff_rows,
    keyset_diff,
    same_bytes,
    stored_form,
)
from training.goldens.compute import compute_obs, compute_supervision
from training.goldens.hashes import ObsDigest
from training.goldens.loaders import load_fixture
from training.goldens.paths import REFERENCES_DIR
from training.goldens.registry import (
    FixtureRecord,
    KeyGroup,
    ObsEntry,
    key_groups,
    obs_entries,
    supervision_entries,
)
from training.goldens.render import render_key_diff, render_keyset_diff
from training.goldens.store import RefId, Surface


class RefStatus(Enum):
    NEW = "new"
    UNCHANGED = "unchanged"
    CHANGED = "changed"
    MOVED = "moved"


@dataclass(frozen=True)
class PlannedObs:
    entry: ObsEntry
    digest: ObsDigest
    status: RefStatus
    diff: ObsMismatch | None = None   # set iff CHANGED
    moved_from: RefId | None = None   # set iff MOVED

    @property
    def ref(self) -> RefId:
        return self.entry.ref


@dataclass(frozen=True)
class PlannedSupervision:
    group: KeyGroup
    fixture: FixtureRecord
    array: np.ndarray                 # stored form
    status: RefStatus
    diff: KeyDiff | None = None       # set iff CHANGED
    moved_from: RefId | None = None   # set iff MOVED

    @property
    def ref(self) -> RefId:
        return self.group.ref(self.fixture)


@dataclass(frozen=True)
class FixtureInfo:
    fixture: FixtureRecord
    T: int
    end_t: int


@dataclass(frozen=True)
class FixturePlan:
    info: FixtureInfo
    obs: list[PlannedObs]
    supervision: list[PlannedSupervision]


# Two dependency groups of one key whose stored arrays are byte-identical on
# every fixture: an over-declared dep, or a fixture coverage gap.
@dataclass(frozen=True)
class IdenticalGroups:
    a: KeyGroup
    b: KeyGroup


@dataclass(frozen=True)
class RegenPlan:
    root: Path                              # the tree this plan was made against
    fixtures: list[FixtureInfo]
    obs: list[PlannedObs]
    supervision: list[PlannedSupervision]
    removed: list[RefId]                    # orphans without matching "plan" content
    byte_matches: list[tuple[RefId, RefId]] # (new file, removed file) with identical content
    unrecognized: list[Path]                # npz / npy files with an invalid path, relative to root
    ignored: list[Path]                     # unexpected non-reference files, relative to root
    warnings: list[IdenticalGroups]

    def is_noop(self) -> bool:
        items = [*self.obs, *self.supervision]
        return all(i.status is RefStatus.UNCHANGED for i in items) and not self.removed

    # Surfaces with changed bytes (a fire), as opposed to new, moved, or removed files.
    def changed_surfaces(self) -> set[Surface]:
        out = set()
        if any(o.status is RefStatus.CHANGED for o in self.obs):
            out.add(Surface.OBS)
        if any(s.status is RefStatus.CHANGED for s in self.supervision):
            out.add(Surface.SUPERVISION)
        return out


class RegenAbort(Exception):
    # Raised during planning before anything is written.
    pass


# --- Per fixture ---


# This is the one planning function bound to the registry (it plans the registry's
# entries for one fixture). If a synthetic-registry test is ever wanted, the entry
# lists can become parameters with the registry as the default, like `root`.
def plan_fixture(fx: FixtureRecord, root: Path = REFERENCES_DIR) -> FixturePlan:
    game, persp = load_fixture(fx)
    return FixturePlan(
        info=FixtureInfo(fixture=fx, T=game.T, end_t=persp.end_t),
        obs=_plan_obs(fx, game, persp, root),
        supervision=_plan_supervision(fx, game, persp, root),
    )


def _plan_obs(
    fx: FixtureRecord, game: SimGame, persp: PerspectiveMeta, root: Path
) -> list[PlannedObs]:
    planned = []
    for e in obs_entries():
        if e.fixture != fx:
            continue
        digest = compute_obs(game, persp, e.point.cfg)
        old = store.load_obs(e.ref, root)
        if old is None:
            planned.append(PlannedObs(entry=e, digest=digest, status=RefStatus.NEW))
            continue
        diff = compare_obs(digest, old)
        status = RefStatus.UNCHANGED if diff is None else RefStatus.CHANGED
        planned.append(PlannedObs(entry=e, digest=digest, status=status, diff=diff))
    return planned


def _plan_supervision(
    fx: FixtureRecord, game: SimGame, persp: PerspectiveMeta, root: Path
) -> list[PlannedSupervision]:
    entries = [e for e in supervision_entries() if e.fixture == fx]
    got = {e.point.name: compute_supervision(game, persp, e.spec) for e in entries}

    for e in entries:
        if (kd := keyset_diff(got[e.point.name], e.point.keys)) is not None:
            raise RegenAbort(f"keyset mismatch for {e.point.name}: {render_keyset_diff(kd)}")

    planned: list[PlannedSupervision] = []
    for group in key_groups():
        key = group.key
        arrays = {p: got[p][key.name] for p in group.points}
        _check_agreement(fx, group, arrays)

        array = stored_form(key.form, arrays[group.representative])
        old = store.load_supervision(group.ref(fx), root)
        if old is None:
            status, diff = RefStatus.NEW, None
        else:
            diff = diff_rows(key.name, array, old)
            status = RefStatus.UNCHANGED if diff is None else RefStatus.CHANGED
        planned.append(
            PlannedSupervision(group=group, fixture=fx, array=array, status=status, diff=diff)
        )
    return planned


# The dependency table claims every point in a group emits the same bytes.
def _check_agreement(fx: FixtureRecord, group: KeyGroup, arrays: dict[str, np.ndarray]) -> None:
    key = group.key
    base = arrays[group.representative]
    agree = [p for p in group.points if same_bytes(arrays[p], base)]
    differ = [p for p in group.points if p not in agree]
    if not differ:
        return
    lines = [
        f"contradiction on fixture {fx.id}: key {key.name!r} deps={key.deps}",
        f"  agree:  {', '.join(agree)}",
    ]
    for p in differ:
        diff = diff_rows(key.name, arrays[p], base)
        assert diff is not None
        lines.append(f"  differs: {p} ({render_key_diff(diff)})")
    lines.append("Either the dependency table is stale or the code diverged.")
    raise RegenAbort("\n".join(lines))


# --- Across fixtures ---


def assemble(parts: list[FixturePlan], root: Path = REFERENCES_DIR) -> RegenPlan:
    obs = [o for p in parts for o in p.obs]
    supervision = [s for p in parts for s in p.supervision]
    warnings = _check_identical_groups(supervision)

    stored = store.build_ref_set(root)
    planned_refs = {o.ref for o in obs} | {s.ref for s in supervision}
    orphans = [r for r in stored.refs if r not in planned_refs]
    obs, supervision, removed, byte_matches = _detect_moves(obs, supervision, orphans, root)

    return RegenPlan(
        root=root,
        fixtures=[p.info for p in parts],
        obs=obs,
        supervision=supervision,
        removed=removed,
        byte_matches=byte_matches,
        unrecognized=stored.unrecognized,
        ignored=stored.ignored,
        warnings=warnings,
    )


# Two groups of a key may agree on one fixture (nothing in it exercises the
# fields they differ on). Agreeing on every fixture is what the warning is for,
# so the check is per key across all fixtures.
def _check_identical_groups(supervision: list[PlannedSupervision]) -> list[IdenticalGroups]:
    # Each group's arrays, in fixture order.
    arrays: dict[KeyGroup, list[np.ndarray]] = defaultdict(list)
    for s in supervision:
        arrays[s.group].append(s.array)
    warnings = []
    for a, b in combinations(arrays, 2):
        if a.key == b.key and all(
            same_bytes(x, y) for x, y in zip(arrays[a], arrays[b], strict=True)
        ):
            warnings.append(IdenticalGroups(a=a, b=b))
    return warnings


def _same_digest(a: ObsDigest, b: ObsDigest) -> bool:
    return same_bytes(a.frame_hashes, b.frame_hashes) and same_bytes(
        a.channel_hashes, b.channel_hashes
    )


# Detect and assign `RefStatus.MOVED`.
# Moved refs are planned refs with a new, currently vacant path that matches an
# orphan on everything but the point (including content). This occurs with point
# renames, or if the representative point changes (supervision only).
# Each orphan is matched at most once. Content that matches an orphan but has a
# different fixture or key is not considered "moved", as two keys can legitimately
# contain identical data. That case is reported as a "byte match" instead.
def _detect_moves(
    obs: list[PlannedObs],
    supervision: list[PlannedSupervision],
    orphans: list[RefId],
    root: Path = REFERENCES_DIR,
) -> tuple[list[PlannedObs], list[PlannedSupervision], list[RefId], list[tuple[RefId, RefId]]]:
    obs_orphans: dict[RefId, ObsDigest] = {}
    sup_orphans: dict[RefId, np.ndarray] = {}
    for rid in orphans:
        if rid.surface is Surface.OBS:
            old_obs = store.load_obs(rid, root)
            assert old_obs is not None, rid
            obs_orphans[rid] = old_obs
        else:
            old_sup = store.load_supervision(rid, root)
            assert old_sup is not None, rid
            sup_orphans[rid] = old_sup

    def same_slot(rid: RefId, fixture: str, key: str | None) -> bool:
        return rid.fixture == fixture and rid.key == key

    out_obs = []
    for o in obs:
        if o.status is RefStatus.NEW:
            for rid, old_obs in obs_orphans.items():
                if same_slot(rid, o.entry.fixture.id, None) and _same_digest(old_obs, o.digest):
                    o = replace(o, status=RefStatus.MOVED, moved_from=rid)
                    del obs_orphans[rid]
                    break
        out_obs.append(o)

    out_sup = []
    for s in supervision:
        if s.status is RefStatus.NEW:
            for rid, old_sup in sup_orphans.items():
                if same_slot(rid, s.fixture.id, s.group.key.name) and same_bytes(old_sup, s.array):
                    s = replace(s, status=RefStatus.MOVED, moved_from=rid)
                    del sup_orphans[rid]
                    break
        out_sup.append(s)

    byte_matches: list[tuple[RefId, RefId]] = []
    for o in out_obs:
        if o.status is RefStatus.NEW:
            for rid, old_obs in obs_orphans.items():
                if _same_digest(old_obs, o.digest):
                    byte_matches.append((o.ref, rid))
    for s in out_sup:
        if s.status is RefStatus.NEW:
            for rid, old_sup in sup_orphans.items():
                if same_bytes(old_sup, s.array):
                    byte_matches.append((s.ref, rid))

    removed = [rid for rid in orphans if rid in obs_orphans or rid in sup_orphans]
    return out_obs, out_sup, removed, byte_matches
