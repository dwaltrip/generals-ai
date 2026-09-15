"""
Produce and write golden references for all registry entries.
Run from packages/training:
    uv run python -m training.goldens.regen
"""

# NOTE: Current status is semi-prototypish, vibe-cded by Claude.

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.bc.datapipe.sim_types import GameMeta
from training.goldens.compare import compare_obs, diff_rows, same_bytes, stored_form
from training.goldens.compute import compute_obs, compute_supervision
from training.goldens.loaders import load_array, load_fixture, load_obs_digest, perspective_for
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


def _status(key: str, old: np.ndarray | None, new: np.ndarray) -> str:
    if old is None:
        return "new"
    diff = diff_rows(key, new, old)
    return "unchanged" if diff is None else f"changed ({diff.summary()})"


def _rel(path: Path) -> str:
    return str(path.relative_to(REFERENCES_DIR))


def _regen_obs(fx: FixtureRecord, sim, game_meta: GameMeta, persp) -> None:
    for entry in (e for e in obs_entries() if e.fixture == fx):
        digest = compute_obs(sim, game_meta, persp, entry.cfg)
        old = load_obs_digest(entry.ref_path)
        if old is None:
            status = "new"
        else:
            mismatch = compare_obs(digest, old)
            status = "unchanged" if mismatch is None else f"changed ({mismatch.summary()})"
        entry.ref_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            entry.ref_path,
            frame_hashes=digest.frame_hashes,
            channel_hashes=digest.channel_hashes,
        )
        print(f"  {status:12s} {_rel(entry.ref_path)}")


@dataclass(frozen=True)
class _PlannedFile:
    path: Path
    array: np.ndarray
    members: tuple[str, ...]


def _plan_supervision(
    fx: FixtureRecord, sim, game_meta: GameMeta, persp
) -> tuple[list[_PlannedFile], list[str]]:
    entries = [e for e in supervision_entries() if e.fixture == fx]
    got = {e.point: compute_supervision(sim, game_meta, persp, e.spec) for e in entries}

    for e in entries:
        if set(got[e.point]) != set(e.keys):
            print(
                f"keyset mismatch for {e.point}: extra {set(got[e.point]) - set(e.keys)},"
                f" missing {set(e.keys) - set(got[e.point])}"
            )
            sys.exit(1)

    planned: list[_PlannedFile] = []
    warnings: list[str] = []
    for key in SUPERVISION_KEYS:
        groups: dict[tuple[Any, ...], list[SupervisionEntry]] = defaultdict(list)
        for e in entries:
            if key.name in e.keys:
                groups[group_id(key, e.spec)].append(e)

        group_arrays: dict[tuple[Any, ...], np.ndarray] = {}
        for gid, members in groups.items():
            base = got[members[0].point][key.name]
            agree = [m.point for m in members if same_bytes(got[m.point][key.name], base)]
            differ = [m.point for m in members if m.point not in agree]
            if differ:
                print(f"\ncontradiction on fixture {fx.id}: key {key.name!r} deps={key.deps}")
                print(f"  agree:  {', '.join(agree)}")
                for p in differ:
                    diff = diff_rows(key.name, got[p][key.name], base)
                    assert diff is not None
                    print(f"  differs: {p} ({diff.summary()})")
                print("Either the dependency table is stale or the code diverged. Nothing written.")
                sys.exit(1)

            stored = stored_form(key.name, base, members[0].hashed_keys)
            group_arrays[gid] = stored
            planned.append(
                _PlannedFile(
                    path=members[0].paths[key.name],
                    array=stored,
                    members=tuple(m.point for m in members),
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


def _write_supervision(fx: FixtureRecord, planned: list[_PlannedFile], warnings: list[str]) -> None:
    fixture_dir = REFERENCES_DIR / "supervision" / fx.id
    fixture_dir.mkdir(parents=True, exist_ok=True)
    written_new: dict[Path, bytes] = {}
    for item in planned:
        old = load_array(item.path)
        status = _status(item.path.stem, old, item.array)
        np.save(item.path, item.array)
        if old is None:
            written_new[item.path] = item.array.tobytes()
        print(f"  {status:12s} {_rel(item.path)}   <- {', '.join(item.members)}")
    for line in warnings:
        print(line)

    expected = {item.path for item in planned}
    for path in sorted(fixture_dir.iterdir()):
        if path in expected:
            continue
        kind = "moved" if any(path.read_bytes() == b for b in written_new.values()) else "removed"
        path.unlink()
        print(f"  {kind:12s} {_rel(path)}")


def main() -> None:
    # Compute and check every fixture before writing anything, so a
    # contradiction leaves the references untouched.
    loaded = []
    for fx in FIXTURES:
        sim, meta = load_fixture(fx)
        game_meta = GameMeta.from_npz(sim, meta)
        persp = perspective_for(game_meta, fx.slot)
        planned, warnings = _plan_supervision(fx, sim, game_meta, persp)
        loaded.append((fx, sim, game_meta, persp, planned, warnings))

    for fx, sim, game_meta, persp, planned, warnings in loaded:
        print(f"== {fx.id}  T={game_meta.T} end_t={persp.end_t}  ({fx.note})")
        _regen_obs(fx, sim, game_meta, persp)
        _write_supervision(fx, planned, warnings)


if __name__ == "__main__":
    main()
