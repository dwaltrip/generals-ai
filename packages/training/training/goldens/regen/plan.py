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
from training.goldens.paths import REFERENCES_DIR
from training.goldens.registry import (
    SUPERVISION_KEYS,
    FixtureRecord,
    GroupId,
    ObsEntry,
    SupervisionEntry,
    SupervisionKey,
    group_id,
    obs_entries,
    supervision_entries,
)
from training.goldens.store import RefId, Surface


class Status(Enum):
    NEW = "new"              # no reference on disk
    UNCHANGED = "unchanged"
    CHANGED = "changed"
    MOVED = "moved"          # new at this path, and an orphan with the same fixture and key
                             # (only the point differs) has identical content


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


# Two dependency groups of one key whose stored arrays are byte-identical on
# every fixture: an over-declared dep, or a fixture coverage gap.
@dataclass(frozen=True)
class IdenticalGroups:
    key: str
    points_a: tuple[str, ...]
    points_b: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    fixtures: list[FixtureInfo]
    obs: list[PlannedObs]
    supervision: list[PlannedSupervision]
    removed: list[RefId]                    # orphans with no matching planned content
    byte_matches: list[tuple[RefId, RefId]] # (new file, removed file) with identical content
    unrecognized: list[Path]                # reference files whose path doesn't parse
    ignored: list[Path]                     # non-reference files on the tree
    warnings: list[IdenticalGroups]


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
            planned.append(PlannedObs(entry=e, digest=digest, status=Status.NEW))
            continue
        diff = compare_obs(digest, old)
        status = Status.UNCHANGED if diff is None else Status.CHANGED
        planned.append(PlannedObs(entry=e, digest=digest, status=status, diff=diff))
    return planned


def _plan_supervision(
    fx: FixtureRecord, game: SimGame, persp: PerspectiveMeta, root: Path
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
            old = store.load_supervision(ref, root)
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


def assemble(parts: list[FixturePlan], root: Path = REFERENCES_DIR) -> Plan:
    obs = [o for p in parts for o in p.obs]
    supervision = [s for p in parts for s in p.supervision]
    warnings = _check_identical_groups([p.supervision for p in parts])

    listing = store.list_refs(root)
    planned_refs = {o.ref for o in obs} | {s.ref for s in supervision}
    orphans = [r for r in listing.refs if r not in planned_refs]
    obs, supervision, removed, byte_matches = _detect_moves(obs, supervision, orphans, root)

    return Plan(
        fixtures=[p.info for p in parts],
        obs=obs,
        supervision=supervision,
        removed=removed,
        byte_matches=byte_matches,
        unrecognized=listing.unrecognized,
        ignored=listing.ignored,
        warnings=warnings,
    )


# Two groups of a key may agree on one fixture (nothing in it exercises the
# fields they differ on). Agreeing on every fixture is what the warning is for,
# so the check is per key across all fixtures.
def _check_identical_groups(per_fixture: list[list[PlannedSupervision]]) -> list[IdenticalGroups]:
    if not per_fixture:
        return []
    warnings = []
    key_names = list(dict.fromkeys(item.key.name for item in per_fixture[0]))
    for key in key_names:
        per_fx = [{f.gid: f for f in items if f.key.name == key} for items in per_fixture]
        gids = list(per_fx[0])
        assert all(set(fx) == set(gids) for fx in per_fx), f"{key}: groups differ by fixture"
        for a, b in combinations(gids, 2):
            if all(same_bytes(fx[a].array, fx[b].array) for fx in per_fx):
                first = per_fx[0]
                warnings.append(
                    IdenticalGroups(key=key, points_a=first[a].points, points_b=first[b].points)
                )
    return warnings


def _same_digest(a: ObsDigest, b: ObsDigest) -> bool:
    return same_bytes(a.frame_hashes, b.frame_hashes) and same_bytes(
        a.channel_hashes, b.channel_hashes
    )


# A new file whose content matches an orphan with the same fixture and key is a
# move: the reference was regenerated identically under a different point name,
# most likely because the representative point changed (points re-ordered or one
# removed). Each orphan is matched at most once. Content that matches an orphan
# with a different fixture or key is not a move, since two keys can legitimately
# hold identical arrays, so it is reported as a byte match instead.
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
        if o.status is Status.NEW:
            for rid, old_obs in obs_orphans.items():
                if same_slot(rid, o.entry.fixture.id, None) and _same_digest(old_obs, o.digest):
                    o = replace(o, status=Status.MOVED, moved_from=rid)
                    del obs_orphans[rid]
                    break
        out_obs.append(o)

    out_sup = []
    for s in supervision:
        if s.status is Status.NEW:
            for rid, old_sup in sup_orphans.items():
                if same_slot(rid, s.fixture.id, s.key.name) and same_bytes(old_sup, s.array):
                    s = replace(s, status=Status.MOVED, moved_from=rid)
                    del sup_orphans[rid]
                    break
        out_sup.append(s)

    byte_matches: list[tuple[RefId, RefId]] = []
    for o in out_obs:
        if o.status is Status.NEW:
            for rid, old_obs in obs_orphans.items():
                if _same_digest(old_obs, o.digest):
                    byte_matches.append((o.ref, rid))
    for s in out_sup:
        if s.status is Status.NEW:
            for rid, old_sup in sup_orphans.items():
                if same_bytes(old_sup, s.array):
                    byte_matches.append((s.ref, rid))

    removed = [rid for rid in orphans if rid in obs_orphans or rid in sup_orphans]
    return out_obs, out_sup, removed, byte_matches
