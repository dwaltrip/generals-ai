"""
Produce and write golden references for all registry entries.
Every fixture is computed and checked before anything is written.

Run from packages/training:
    uv run python -m training.goldens.regen
"""

# TODO: still to build: the fire report, choosing the baseline to diff against,
# and the boundary-record skeleton.

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import sys

import numpy as np

from training.bc.datapipe.sim_types import PerspectiveMeta, SimGame
from training.goldens import store
from training.goldens.compare import compare_obs, diff_rows, keyset_diff, same_bytes, stored_form
from training.goldens.compute import compute_obs, compute_supervision
from training.goldens.hashes import ObsDigest
from training.goldens.loaders import load_fixture
from training.goldens.registry import (
    FIXTURES,
    SUPERVISION_KEYS,
    FixtureRecord,
    GroupId,
    SupervisionEntry,
    group_id,
    obs_entries,
    representative,
    supervision_entries,
)
from training.goldens.store import RefId, Surface


@dataclass(frozen=True)
class _ObsFile:
    ref: RefId
    digest: ObsDigest


@dataclass(frozen=True)
class _SupervisionFile:
    # One (key, group) on one fixture.
    ref: RefId
    array: np.ndarray
    key: str
    gid: GroupId
    members: tuple[str, ...]


@dataclass(frozen=True)
class _FixturePlan:
    fixture: FixtureRecord
    T: int
    end_t: int
    obs: list[_ObsFile]
    supervision: list[_SupervisionFile]


class RegenAbort(Exception):
    # Raised during planning before anything is written.
    pass


# --- Plan ---


def _plan_obs(fx: FixtureRecord, game: SimGame, persp: PerspectiveMeta) -> list[_ObsFile]:
    return [
        _ObsFile(ref=e.ref, digest=compute_obs(game, persp, e.point.cfg))
        for e in obs_entries()
        if e.fixture == fx
    ]


def _plan_supervision(
    fx: FixtureRecord, game: SimGame, persp: PerspectiveMeta
) -> list[_SupervisionFile]:
    entries = [e for e in supervision_entries() if e.fixture == fx]
    got = {e.point.name: compute_supervision(game, persp, e.spec) for e in entries}

    for e in entries:
        if (kd := keyset_diff(got[e.point.name], e.point.keys)) is not None:
            raise RegenAbort(f"keyset mismatch for {e.point.name}: {kd.summary()}")

    planned: list[_SupervisionFile] = []
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

            planned.append(
                _SupervisionFile(
                    ref=members[0].refs[key.name].ref,
                    array=stored_form(key.form, base),
                    key=key.name,
                    gid=gid,
                    members=tuple(names),
                )
            )
    return planned


def _plan_fixture(fx: FixtureRecord) -> _FixturePlan:
    game, persp = load_fixture(fx)
    return _FixturePlan(
        fixture=fx,
        T=game.T,
        end_t=persp.end_t,
        obs=_plan_obs(fx, game, persp),
        supervision=_plan_supervision(fx, game, persp),
    )


# Two groups of a key may agree on one fixture (nothing in it exercises the
# fields they differ on). Agreeing on every fixture means either an over-declared
# dep or a fixture coverage gap, so the check is per key across all fixtures.
def _check_identical_groups(plans: list[_FixturePlan]) -> list[str]:
    warnings = []
    for key in SUPERVISION_KEYS:
        per_fx = [{f.gid: f.array for f in p.supervision if f.key == key.name} for p in plans]
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


# --- Write ---


def _write_obs(files: list[_ObsFile]) -> None:
    for item in files:
        old = store.load_obs(item.ref)
        if old is None:
            status = "new"
        else:
            mismatch = compare_obs(item.digest, old)
            status = "unchanged" if mismatch is None else f"changed ({mismatch.summary()})"
        store.save_obs(item.ref, item.digest)
        print(f"  {status:12s} {store.rel(item.ref)}")


# Returns the "new" arrays (references blessed for the first time)
# TODO: per-file status (new / unchanged / changed) is decided at write time.
# Deciding it during planning would leave the writers as pure save-and-print.
# Consider with the fire report.
def _write_supervision(files: list[_SupervisionFile]) -> list[np.ndarray]:
    written_new = []
    for item in files:
        old = store.load_supervision(item.ref)
        if old is None:
            status = "new"
            written_new.append(item.array)
        else:
            diff = diff_rows(item.key, item.array, old)
            status = "unchanged" if diff is None else f"changed ({diff.summary()})"
        store.save_supervision(item.ref, item.array)
        print(f"  {status:12s} {store.rel(item.ref)}   <- {', '.join(item.members)}")
    return written_new


def _remove_orphans(expected: set[RefId], written_new: list[np.ndarray]) -> None:
    # "moved" means the reference was removed and then regenerated identically
    # with a different filepath. Most likely, the representative point changed,
    # e.g. the points were re-ordered or one was removed.
    # NOTE: There are other less common scenarios that can also cause this.
    listing = store.list_refs()
    for rid in listing.refs:
        if rid in expected:
            continue
        moved = False
        if rid.surface is Surface.SUPERVISION:
            old = store.load_supervision(rid)
            assert old is not None
            moved = any(same_bytes(old, a) for a in written_new)
        store.remove(rid)
        print(f"  {'moved' if moved else 'removed':12s} {store.rel(rid)}")
    for path in listing.unrecognized:
        path.unlink()
        print(f"  {'removed':12s} {path.name} (unrecognized file)")


def main() -> int:
    try:
        plans = [_plan_fixture(fx) for fx in FIXTURES]
    except RegenAbort as e:
        print(f"\n{e}\nNothing written.")
        return 1

    written_new: list[np.ndarray] = []
    for plan in plans:
        fx = plan.fixture
        print(f"== {fx.id}  T={plan.T} end_t={plan.end_t}  ({fx.note})")
        _write_obs(plan.obs)
        written_new += _write_supervision(plan.supervision)

    for line in _check_identical_groups(plans):
        print(line)

    expected = {f.ref for p in plans for f in p.obs}
    expected |= {f.ref for p in plans for f in p.supervision}
    _remove_orphans(expected, written_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
