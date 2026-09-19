"""
Planning for regen: compute every reference, classify each against the tree
on disk, and detect moves and orphans. Nothing here writes or prints.
"""

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
from training.goldens.registry import (
    SUPERVISION_KEYS,
    FixtureRecord,
    GroupId,
    ObsEntry,
    SupervisionEntry,
    SupervisionKey,
    group_id,
    obs_entries,
    representative,
    supervision_entries,
)
from training.goldens.store import RefId, Surface


class Status(Enum):
    NEW = "new"              # no reference on disk
    UNCHANGED = "unchanged"
    CHANGED = "changed"
    MOVED = "moved"          # new at this path, and an orphan elsewhere has identical content


@dataclass(frozen=True)
class PlannedObs:
    entry: ObsEntry
    digest: ObsDigest
    status: Status
    diff: ObsMismatch | None = None   # set iff CHANGED
    moved_from: RefId | None = None   # set iff MOVED

    @property
    def ref(self) -> RefId:
        return self.entry.ref


@dataclass(frozen=True)
class PlannedSupervision:
    # One (key, group) on one fixture.
    ref: RefId
    key: SupervisionKey
    fixture: FixtureRecord
    gid: GroupId
    points: tuple[str, ...]           # the points sharing this file, registry order
    array: np.ndarray                 # stored form
    status: Status
    diff: KeyDiff | None = None       # set iff CHANGED
    moved_from: RefId | None = None   # set iff MOVED


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


@dataclass(frozen=True)
class Plan:
    fixtures: list[FixtureInfo]
    obs: list[PlannedObs]
    supervision: list[PlannedSupervision]
    removed: list[RefId]        # orphans with no matching planned content
    unrecognized: list[Path]    # files on the tree that don't parse
    warnings: list[str]


class RegenAbort(Exception):
    # Raised during planning before anything is written.
    pass


# --- Per fixture ---


def plan_fixture(fx: FixtureRecord) -> FixturePlan:
    game, persp = load_fixture(fx)
    return FixturePlan(
        info=FixtureInfo(fixture=fx, T=game.T, end_t=persp.end_t),
        obs=_plan_obs(fx, game, persp),
        supervision=_plan_supervision(fx, game, persp),
    )


def _plan_obs(fx: FixtureRecord, game: SimGame, persp: PerspectiveMeta) -> list[PlannedObs]:
    planned = []
    for e in obs_entries():
        if e.fixture != fx:
            continue
        digest = compute_obs(game, persp, e.point.cfg)
        old = store.load_obs(e.ref)
        if old is None:
            planned.append(PlannedObs(entry=e, digest=digest, status=Status.NEW))
            continue
        diff = compare_obs(digest, old)
        status = Status.UNCHANGED if diff is None else Status.CHANGED
        planned.append(PlannedObs(entry=e, digest=digest, status=status, diff=diff))
    return planned


def _plan_supervision(
    fx: FixtureRecord, game: SimGame, persp: PerspectiveMeta
) -> list[PlannedSupervision]:
    entries = [e for e in supervision_entries() if e.fixture == fx]
    got = {e.point.name: compute_supervision(game, persp, e.spec) for e in entries}

    for e in entries:
        if (kd := keyset_diff(got[e.point.name], e.point.keys)) is not None:
            raise RegenAbort(f"keyset mismatch for {e.point.name}: {kd.summary()}")

    planned: list[PlannedSupervision] = []
    for key in SUPERVISION_KEYS:
        groups: dict[GroupId, list[SupervisionEntry]] = defaultdict(list)
        for e in entries:
            if key.name in e.point.keys:
                groups[group_id(key, e.point.cfg)].append(e)

        for gid, members in groups.items():
            names = [m.point.name for m in members]
            base = got[names[0]][key.name]
            agree = [n for n in names if same_bytes(got[n][key.name], base)]
            differ = [n for n in names if n not in agree]
            if differ:
                lines = [
                    f"contradiction on fixture {fx.id}: key {key.name!r} deps={key.deps}",
                    f"  agree:  {', '.join(agree)}",
                ]
                for p in differ:
                    diff = diff_rows(key.name, got[p][key.name], base)
                    assert diff is not None
                    lines.append(f"  differs: {p} ({diff.summary()})")
                lines.append("Either the dependency table is stale or the code diverged.")
                raise RegenAbort("\n".join(lines))

            ref = members[0].refs[key.name].ref
            array = stored_form(key.form, base)
            old = store.load_supervision(ref)
            if old is None:
                status, diff = Status.NEW, None
            else:
                diff = diff_rows(key.name, array, old)
                status = Status.UNCHANGED if diff is None else Status.CHANGED
            planned.append(
                PlannedSupervision(
                    ref=ref,
                    key=key,
                    fixture=fx,
                    gid=gid,
                    points=tuple(names),
                    array=array,
                    status=status,
                    diff=diff,
                )
            )
    return planned


# --- Across fixtures ---


def assemble(parts: list[FixturePlan]) -> Plan:
    obs = [o for p in parts for o in p.obs]
    supervision = [s for p in parts for s in p.supervision]
    warnings = _check_identical_groups([p.supervision for p in parts])

    listing = store.list_refs()
    planned_refs = {o.ref for o in obs} | {s.ref for s in supervision}
    orphans = [r for r in listing.refs if r not in planned_refs]
    obs, supervision, removed = _detect_moves(obs, supervision, orphans)

    return Plan(
        fixtures=[p.info for p in parts],
        obs=obs,
        supervision=supervision,
        removed=removed,
        unrecognized=listing.unrecognized,
        warnings=warnings,
    )


# Two groups of a key may agree on one fixture (nothing in it exercises the
# fields they differ on). Agreeing on every fixture means either an over-declared
# dep or a fixture coverage gap, so the check is per key across all fixtures.
def _check_identical_groups(per_fixture: list[list[PlannedSupervision]]) -> list[str]:
    warnings = []
    for key in SUPERVISION_KEYS:
        per_fx = [{f.gid: f.array for f in items if f.key == key} for items in per_fixture]
        gids = list(per_fx[0])
        assert all(set(fx) == set(gids) for fx in per_fx), f"{key.name}: groups differ by fixture"
        for a, b in combinations(gids, 2):
            if all(same_bytes(fx[a], fx[b]) for fx in per_fx):
                warnings.append(
                    f"  warning: {key.name}: groups {representative(key, a)} and"
                    f" {representative(key, b)} are byte-identical on every fixture"
                    " (over-declared deps, or a fixture coverage gap)"
                )
    return warnings


def _same_digest(a: ObsDigest, b: ObsDigest) -> bool:
    return same_bytes(a.frame_hashes, b.frame_hashes) and same_bytes(
        a.channel_hashes, b.channel_hashes
    )


# A planned file with no reference at its path, whose content matches an orphan
# elsewhere, is a move: the reference was regenerated identically under a new
# path. Most likely the representative point changed (points re-ordered or one
# removed). Each orphan is matched at most once.
def _detect_moves(
    obs: list[PlannedObs], supervision: list[PlannedSupervision], orphans: list[RefId]
) -> tuple[list[PlannedObs], list[PlannedSupervision], list[RefId]]:
    pending = list(orphans)

    def take(surface: Surface, matches) -> RefId | None:
        for rid in pending:
            if rid.surface is surface and matches(rid):
                pending.remove(rid)
                return rid
        return None

    def loaded_obs(rid: RefId) -> ObsDigest:
        old = store.load_obs(rid)
        assert old is not None, rid
        return old

    def loaded_supervision(rid: RefId) -> np.ndarray:
        old = store.load_supervision(rid)
        assert old is not None, rid
        return old

    out_obs = []
    for o in obs:
        if o.status is Status.NEW:
            src = take(Surface.OBS, lambda rid, o=o: _same_digest(loaded_obs(rid), o.digest))
            if src is not None:
                o = replace(o, status=Status.MOVED, moved_from=src)
        out_obs.append(o)

    out_sup = []
    for s in supervision:
        if s.status is Status.NEW:
            src = take(
                Surface.SUPERVISION, lambda rid, s=s: same_bytes(loaded_supervision(rid), s.array)
            )
            if src is not None:
                s = replace(s, status=Status.MOVED, moved_from=src)
        out_sup.append(s)

    return out_obs, out_sup, pending
