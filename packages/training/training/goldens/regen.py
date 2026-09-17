"""
Produce and write golden references for all registry entries.
Run from packages/training:
    uv run python -m training.goldens.regen

Every fixture is computed and checked before anything is written, so a
contradiction leaves the references untouched. Afterward, files under the
references tree that no entry claims are deleted and reported.

TODO: still to build: the fire report beyond per-file status lines, choosing
the baseline to diff against, the boundary-record skeleton, and the
identical-groups check done once per key across all fixtures (today it runs
per fixture, which is the wrong granularity: see 9.13-1).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.bc.datapipe.sim_types import PerspectiveMeta, SimGame
from training.goldens.compare import compare_obs, diff_rows, keyset_diff, same_bytes, stored_form
from training.goldens.compute import compute_obs, compute_supervision
from training.goldens.hashes import ObsDigest
from training.goldens.loaders import load_array, load_fixture, load_obs_digest
from training.goldens.registry import (
    FIXTURES,
    REFERENCES_DIR,
    SUPERVISION_KEYS,
    FixtureRecord,
    SupervisionEntry,
    group_id,
    obs_entries,
    supervision_entries,
)


@dataclass(frozen=True)
class _ObsFile:
    path: Path
    digest: ObsDigest


@dataclass(frozen=True)
class _SupervisionFile:
    path: Path
    array: np.ndarray
    members: tuple[str, ...]


@dataclass(frozen=True)
class _FixturePlan:
    fixture: FixtureRecord
    T: int
    end_t: int
    obs: list[_ObsFile]
    supervision: list[_SupervisionFile]
    warnings: list[str]


class RegenAbort(Exception):
    # Raised during planning. The message is the report; nothing has been written.
    pass


def _rel(path: Path) -> str:
    return str(path.relative_to(REFERENCES_DIR))


# --- Plan ---


def _plan_obs(fx: FixtureRecord, game: SimGame, persp: PerspectiveMeta) -> list[_ObsFile]:
    return [
        _ObsFile(path=e.ref_path, digest=compute_obs(game, persp, e.point.cfg))
        for e in obs_entries()
        if e.fixture == fx
    ]


def _plan_supervision(
    fx: FixtureRecord, game: SimGame, persp: PerspectiveMeta
) -> tuple[list[_SupervisionFile], list[str]]:
    entries = [e for e in supervision_entries() if e.fixture == fx]
    got = {e.point.name: compute_supervision(game, persp, e.spec) for e in entries}

    for e in entries:
        if (kd := keyset_diff(got[e.point.name], e.point.keys)) is not None:
            raise RegenAbort(f"keyset mismatch for {e.point.name}: {kd.summary()}")

    planned: list[_SupervisionFile] = []
    # TODO: warnings are preformatted strings built here and printed by main,
    # so presentation leaks into planning. Revisit with the second-pass redesign
    # of this check (see the module docstring).
    warnings: list[str] = []
    for key in SUPERVISION_KEYS:
        groups: dict[tuple[Any, ...], list[SupervisionEntry]] = defaultdict(list)
        for e in entries:
            if key.name in e.point.keys:
                groups[group_id(key, e.point.cfg)].append(e)

        group_arrays: dict[tuple[Any, ...], np.ndarray] = {}
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

            stored = stored_form(key.form, base)
            group_arrays[gid] = stored
            planned.append(
                _SupervisionFile(
                    path=members[0].refs[key.name].path,
                    array=stored,
                    members=tuple(names),
                )
            )

        gids = list(group_arrays)
        for i in range(len(gids)):
            for j in range(i + 1, len(gids)):
                if same_bytes(group_arrays[gids[i]], group_arrays[gids[j]]):
                    warnings.append(
                        f"  warning: {key.name} groups {gids[i]} and {gids[j]} are byte-identical"
                        " on this fixture (over-declared deps, or unexercised)"
                    )
    return planned, warnings


def _plan_fixture(fx: FixtureRecord) -> _FixturePlan:
    game, persp = load_fixture(fx)
    supervision, warnings = _plan_supervision(fx, game, persp)
    return _FixturePlan(
        fixture=fx,
        T=game.T,
        end_t=persp.end_t,
        obs=_plan_obs(fx, game, persp),
        supervision=supervision,
        warnings=warnings,
    )


# --- Write ---


def _write_obs(files: list[_ObsFile]) -> None:
    for item in files:
        old = load_obs_digest(item.path)
        if old is None:
            status = "new"
        else:
            mismatch = compare_obs(item.digest, old)
            status = "unchanged" if mismatch is None else f"changed ({mismatch.summary()})"
        item.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            item.path,
            frame_hashes=item.digest.frame_hashes,
            channel_hashes=item.digest.channel_hashes,
        )
        print(f"  {status:12s} {_rel(item.path)}")


# Returns the "new" arrays (references blessed for the first time)
# TODO: the per-file status (new / unchanged / changed) is decided here at write
# time, where the old file is last observable. It could be decided during
# planning instead, which would leave the writers as pure save-and-print.
# Consider with the fire report (see the module docstring).
def _write_supervision(files: list[_SupervisionFile]) -> list[np.ndarray]:
    written_new = []
    for item in files:
        old = load_array(item.path)
        if old is None:
            status = "new"
            written_new.append(item.array)
        else:
            diff = diff_rows(item.path.stem, item.array, old)
            status = "unchanged" if diff is None else f"changed ({diff.summary()})"
        item.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(item.path, item.array)
        print(f"  {status:12s} {_rel(item.path)}   <- {', '.join(item.members)}")
    return written_new


def _remove_orphans(expected: set[Path], written_new: list[np.ndarray]) -> None:
    # A supervision orphan whose contents match a file written as new this run
    # is a renamed group file (its representative point changed), reported as
    # "moved". Anything else is "removed".
    for path in sorted(p for p in REFERENCES_DIR.rglob("*") if p.is_file()):
        if path in expected or path.name.startswith("."):
            continue
        moved = path.suffix == ".npy" and any(same_bytes(np.load(path), a) for a in written_new)
        path.unlink()
        print(f"  {'moved' if moved else 'removed':12s} {_rel(path)}")


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
        for line in plan.warnings:
            print(line)

    expected = {f.path for p in plans for f in p.obs}
    expected |= {f.path for p in plans for f in p.supervision}
    _remove_orphans(expected, written_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
