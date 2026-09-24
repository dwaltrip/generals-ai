"""Build a regen report from a RegenPlan."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from training.goldens import store
from training.goldens.compare import KeyDiff, ObsMismatch
from training.goldens.regen.plan import (
    IdenticalGroups,
    PlannedObs,
    PlannedSupervision,
    RefStatus,
    RegenPlan,
)
from training.goldens.registry import KeyGroup
from training.goldens.store import RefId, Surface


_W_NAME = 24   # point or key column
_W_FX = 16     # fixture column
_W_CH = 30     # channel column


def render_regen_report(plan: RegenPlan) -> str:
    lines = ["regen: compared against the references on disk before this run", ""]
    lines.append(_headline(plan))
    lines.append("")
    lines += _warnings(plan)
    lines.append("")
    if not plan.is_noop():
        lines += _obs_section(plan)
        lines.append("")
        lines += _supervision_section(plan)
        lines.append("")
    lines += _stray_files(plan)
    return "\n".join(lines).rstrip() + "\n"


# --- Headline and warnings ---


def _headline(plan: RegenPlan) -> str:
    counts = Counter(i.status for i in [*plan.obs, *plan.supervision])
    if plan.is_noop():
        return f"no changes   unchanged {counts[RefStatus.UNCHANGED]}"
    parts = [f"{s.value} {counts[s]}" for s in (RefStatus.CHANGED, RefStatus.NEW, RefStatus.MOVED)]
    parts.append(f"removed {len(plan.removed)}")
    parts.append(f"unchanged {counts[RefStatus.UNCHANGED]}")
    return "   ".join(parts)


def _warnings(plan: RegenPlan) -> list[str]:
    if not plan.warnings:
        return ["warnings: none"]
    return ["warnings:"] + [f"  {render_warning(w)}" for w in plan.warnings]


def render_warning(w: IdenticalGroups) -> str:
    return (
        f"{w.a.key.name}: groups {w.a.representative} and {w.b.representative} are byte-identical"
        " on every fixture (over-declared deps, or a fixture coverage gap)"
    )


# --- Per-fixture span ---


# Description of the changed ticks, along an axis of length `total`.
def _ticks(changed: np.ndarray, total: int) -> str:
    k = int(changed.size)
    if k == 0:
        return "unchanged"
    if k == total:
        return f"all {total}"
    first, last = int(changed[0]), int(changed[-1])
    if k == 1:
        return f"t={first}"
    if last - first + 1 == k:
        return f"t={first}..{last}"
    return f"{k} of {total}, from t={first}"


def _obs_frames(d: ObsMismatch) -> str:
    if d.changed_frames is None:
        return f"frame count {d.ref.n_frames} to {d.got.n_frames}"
    return _ticks(d.changed_frames, d.ref.n_frames)


def _rows(d: KeyDiff) -> str:
    if d.changed_rows is None:
        return _render_layout_change(d)
    return _ticks(d.changed_rows, d.ref.shape[0])


def _render_layout_change(d: KeyDiff) -> str:
    ref, got = d.ref, d.got
    if ref.shape == got.shape:
        return f"dtype {ref.dtype} to {got.dtype}"
    if ref.dtype == got.dtype and ref.shape[1:] == got.shape[1:]:
        return f"row count {ref.shape[0]} to {got.shape[0]}"
    return f"layout {ref.shape} {ref.dtype} to {got.shape} {got.dtype}"


# --- Obs, pivoted by point ---


def _obs_section(plan: RegenPlan) -> list[str]:
    lines = ["obs"]
    by_point: dict[str, list[PlannedObs]] = defaultdict(list)
    for item in plan.obs:
        by_point[item.entry.point.name].append(item)
    for point, items in by_point.items():
        lines += _obs_point(point, items)

    removed_by_point: dict[str, list[RefId]] = defaultdict(list)
    for rid in plan.removed:
        if rid.surface is Surface.OBS:
            removed_by_point[rid.point].append(rid)
    for point, rids in removed_by_point.items():
        lines.append(f"  {point:<{_W_NAME}} removed   ({_n_fixtures(len(rids))})")

    lines += _byte_matches(plan, Surface.OBS)
    return lines


def _obs_point(point: str, items: list[PlannedObs]) -> list[str]:
    n = len(items)
    counts = Counter(i.status for i in items)
    if counts[RefStatus.UNCHANGED] == n:
        return [f"  {point:<{_W_NAME}} unchanged"]
    if counts[RefStatus.NEW] == n:
        return [f"  {point:<{_W_NAME}} new, no baseline   ({_n_fixtures(n)})"]
    if counts[RefStatus.MOVED] == n:
        sources = sorted({i.moved_from.point for i in items if i.moved_from is not None})
        return [f"  {point:<{_W_NAME}} moved from {', '.join(sources)}   ({_n_fixtures(n)})"]

    head = []
    if counts[RefStatus.CHANGED]:
        head.append(f"changed on {counts[RefStatus.CHANGED]} of {n} fixtures")
    for s in (RefStatus.NEW, RefStatus.MOVED, RefStatus.UNCHANGED):
        if counts[s] and counts[s] != n:
            head.append(f"{s.value} on {counts[s]}")
    lines = [f"  {point:<{_W_NAME}} {', '.join(head)}"]

    changed = [i for i in items if i.status is RefStatus.CHANGED]
    if changed:
        lines += _obs_channels(changed)
        lines.append("    frames")
        for i in items:
            lines.append(f"      {i.entry.fixture.id:<{_W_FX}} {_obs_status(i)}")
    else:
        for i in items:
            if i.status is not RefStatus.UNCHANGED:
                lines.append(f"    {i.entry.fixture.id:<{_W_FX}} {_obs_status(i)}")
    return lines


def _obs_status(i: PlannedObs) -> str:
    match i.status:
        case RefStatus.CHANGED:
            assert i.diff is not None
            return _obs_frames(i.diff)
        case RefStatus.MOVED:
            assert i.moved_from is not None
            return f"moved from {store.rel_path(i.moved_from)}"
        case RefStatus.NEW:
            return "new, no baseline"
        case RefStatus.UNCHANGED:
            return "unchanged"


def _obs_channels(changed: list[PlannedObs]) -> list[str]:
    lines = ["    channels"]
    # Channel names come from each point's own config. All items here share a
    # point, so one config serves, and the index is shown alongside the name.
    names = changed[0].entry.point.cfg.channel_names
    per_channel: dict[int, list[str]] = defaultdict(list)
    for i in changed:
        assert i.diff is not None
        if i.diff.changed_channels is None:
            old = i.diff.ref.n_channels
            new = i.diff.got.n_channels
            lines.append(
                f"      layout changed, {old} to {new} channels, no per-channel diff"
                f"   ({i.entry.fixture.id})"
            )
            continue
        for c in i.diff.changed_channels.tolist():
            per_channel[c].append(i.entry.fixture.id)
    for c in sorted(per_channel):
        fixtures = per_channel[c]
        label = f"[{c}] {names[c]}"
        tag = _n_fixtures(len(fixtures))
        if len(fixtures) < len(changed):
            tag += f"   ({', '.join(fixtures)})"
        lines.append(f"      {label:<{_W_CH}} {tag}")
    return lines


# --- Supervision, pivoted by key ---


def _supervision_section(plan: RegenPlan) -> list[str]:
    # One block per key group, which is one file per fixture. Groups whose
    # files are all unchanged are omitted.
    blocks = []
    by_group: dict[KeyGroup, list[PlannedSupervision]] = defaultdict(list)
    for item in plan.supervision:
        by_group[item.group].append(item)
    for group, items in by_group.items():
        if all(i.status is RefStatus.UNCHANGED for i in items):
            continue
        blocks.append(f"  {group.key.name:<{_W_NAME}} {', '.join(group.points)}")
        for i in items:
            blocks.append(f"    {i.fixture.id:<{_W_FX}} {_supervision_status(i)}")

    removed_by_key: dict[tuple[str, str], list[RefId]] = defaultdict(list)
    for rid in plan.removed:
        if rid.surface is Surface.SUPERVISION:
            assert rid.key is not None
            removed_by_key[(rid.key, rid.point)].append(rid)
    for (key, point), rids in removed_by_key.items():
        blocks.append(f"  {key:<{_W_NAME}} removed   ({point}, {_n_fixtures(len(rids))})")

    blocks += _byte_matches(plan, Surface.SUPERVISION)

    # The header names the points that fired. The "unchanged" list means no byte
    # changes, so it is shown only when nothing below reports a new, moved, or
    # removed file, where it would read as a contradiction.
    changed_points = _points_with(plan.supervision, RefStatus.CHANGED)
    bytes_only = (
        all(i.status in (RefStatus.CHANGED, RefStatus.UNCHANGED) for i in plan.supervision)
        and not removed_by_key
        and not any(n.surface is Surface.SUPERVISION for n, _ in plan.byte_matches)
    )
    head = "supervision"
    if changed_points:
        head += f"   changed on: {', '.join(changed_points)}"
    if bytes_only:
        quiet = [p for p in _points_with(plan.supervision, None) if p not in changed_points]
        if quiet:
            head += f"   unchanged: {', '.join(quiet)}"
    return [head, *blocks]


def _supervision_status(i: PlannedSupervision) -> str:
    match i.status:
        case RefStatus.CHANGED:
            assert i.diff is not None
            return _rows(i.diff)
        case RefStatus.MOVED:
            assert i.moved_from is not None
            return f"moved from {store.rel_path(i.moved_from)}"
        case RefStatus.NEW:
            return "new, no baseline"
        case RefStatus.UNCHANGED:
            return "unchanged"


def _points_with(items: list[PlannedSupervision], status: RefStatus | None) -> list[str]:
    seen: dict[str, None] = {}
    for i in items:
        if status is None or i.status is status:
            for p in i.group.points:
                seen.setdefault(p, None)
    return list(seen)


# --- Shared ---


def _byte_matches(plan: RegenPlan, surface: Surface) -> list[str]:
    return [
        f"  note: new {store.rel_path(new)} has identical bytes to removed {store.rel_path(old)}"
        for new, old in plan.byte_matches
        if new.surface is surface
    ]


def _stray_files(plan: RegenPlan) -> list[str]:
    lines = [f"unrecognized reference path, left in place: {p}" for p in plan.unrecognized]
    if plan.ignored:
        names = ", ".join(str(p) for p in plan.ignored)
        lines.append(f"ignored {len(plan.ignored)} non-reference file(s): {names}")
    return lines


def _n_fixtures(n: int) -> str:
    return f"{n} fixture" if n == 1 else f"{n} fixtures"
