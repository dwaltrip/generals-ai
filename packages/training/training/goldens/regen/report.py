"""
Renderers over a Plan. Pure functions from the plan to text.

The regen report (9.18-1 section 8): summary first, warnings near the top, then
obs pivoted by point and supervision pivoted by key. Unchanged obs points get
one line each, unchanged supervision keys are omitted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from training.goldens import store
from training.goldens.compare import KeyDiff, ObsMismatch
from training.goldens.regen.plan import (
    IdenticalGroups,
    Plan,
    PlannedObs,
    PlannedSupervision,
    Status,
)
from training.goldens.store import RefId, Surface


_W_NAME = 24   # point or key column
_W_FX = 16     # fixture column
_W_CH = 30     # channel column


def render_report(plan: Plan, root: Path) -> str:
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
    lines += _stray_files(plan, root)
    return "\n".join(lines).rstrip() + "\n"


# --- Headline and warnings ---


def _headline(plan: Plan) -> str:
    counts = Counter(i.status for i in [*plan.obs, *plan.supervision])
    if plan.is_noop():
        return f"no changes   unchanged {counts[Status.UNCHANGED]}"
    parts = [f"{s.value} {counts[s]}" for s in (Status.CHANGED, Status.NEW, Status.MOVED)]
    parts.append(f"removed {len(plan.removed)}")
    parts.append(f"unchanged {counts[Status.UNCHANGED]}")
    return "   ".join(parts)


def _warnings(plan: Plan) -> list[str]:
    if not plan.warnings:
        return ["warnings: none"]
    return ["warnings:"] + [f"  {render_warning(w)}" for w in plan.warnings]


def render_warning(w: IdenticalGroups) -> str:
    return (
        f"{w.key}: groups {w.points_a[0]} and {w.points_b[0]} are byte-identical"
        " on every fixture (over-declared deps, or a fixture coverage gap)"
    )


# --- Per-fixture span ---


# The four-case rendering of changed indices, plus the case where the axis
# length itself changed.
def _span(changed: np.ndarray | None, total: int, counts: tuple[int, int], what: str) -> str:
    if changed is None:
        return f"{what} count {counts[0]} to {counts[1]}"
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
    return _span(d.changed_frames, d.n_frames[1], d.n_frames, "frame")


def _rows(d: KeyDiff) -> str:
    if d.note:
        return d.note
    return _span(d.changed_rows, d.total_rows, (d.total_rows, d.total_rows), "row")


# --- Obs, pivoted by point ---


def _obs_section(plan: Plan) -> list[str]:
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
    if counts[Status.UNCHANGED] == n:
        return [f"  {point:<{_W_NAME}} unchanged"]
    if counts[Status.NEW] == n:
        return [f"  {point:<{_W_NAME}} new, no baseline   ({_n_fixtures(n)})"]
    if counts[Status.MOVED] == n:
        sources = sorted({i.moved_from.point for i in items if i.moved_from is not None})
        return [f"  {point:<{_W_NAME}} moved from {', '.join(sources)}   ({_n_fixtures(n)})"]

    head = []
    if counts[Status.CHANGED]:
        head.append(f"changed on {counts[Status.CHANGED]} of {n} fixtures")
    for s in (Status.NEW, Status.MOVED, Status.UNCHANGED):
        if counts[s] and counts[s] != n:
            head.append(f"{s.value} on {counts[s]}")
    lines = [f"  {point:<{_W_NAME}} {', '.join(head)}"]

    changed = [i for i in items if i.status is Status.CHANGED]
    if changed:
        lines += _obs_channels(changed)
        lines.append("    frames")
        for i in items:
            lines.append(f"      {i.entry.fixture.id:<{_W_FX}} {_obs_status(i)}")
    else:
        for i in items:
            if i.status is not Status.UNCHANGED:
                lines.append(f"    {i.entry.fixture.id:<{_W_FX}} {_obs_status(i)}")
    return lines


def _obs_status(i: PlannedObs) -> str:
    match i.status:
        case Status.CHANGED:
            assert i.diff is not None
            return _obs_frames(i.diff)
        case Status.MOVED:
            assert i.moved_from is not None
            return f"moved from {store.rel(i.moved_from)}"
        case Status.NEW:
            return "new, no baseline"
        case Status.UNCHANGED:
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
            old, new = i.diff.n_channels
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


def _supervision_section(plan: Plan) -> list[str]:
    # One block per (key, group), which is one file per fixture. Keys whose
    # files are all unchanged are omitted.
    blocks = []
    by_file: dict[tuple[str, tuple[str, ...]], list[PlannedSupervision]] = defaultdict(list)
    for item in plan.supervision:
        by_file[(item.key.name, item.points)].append(item)
    for (key, points), items in by_file.items():
        if all(i.status is Status.UNCHANGED for i in items):
            continue
        blocks.append(f"  {key:<{_W_NAME}} {', '.join(points)}")
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
    changed_points = _points_with(plan.supervision, Status.CHANGED)
    bytes_only = (
        all(i.status in (Status.CHANGED, Status.UNCHANGED) for i in plan.supervision)
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
        case Status.CHANGED:
            assert i.diff is not None
            return _rows(i.diff)
        case Status.MOVED:
            assert i.moved_from is not None
            return f"moved from {store.rel(i.moved_from)}"
        case Status.NEW:
            return "new, no baseline"
        case Status.UNCHANGED:
            return "unchanged"


def _points_with(items: list[PlannedSupervision], status: Status | None) -> list[str]:
    seen: dict[str, None] = {}
    for i in items:
        if status is None or i.status is status:
            for p in i.points:
                seen.setdefault(p, None)
    return list(seen)


# --- Shared ---


def _byte_matches(plan: Plan, surface: Surface) -> list[str]:
    return [
        f"  note: new {store.rel(new)} has identical bytes to removed {store.rel(old)}"
        for new, old in plan.byte_matches
        if new.surface is surface
    ]


def _stray_files(plan: Plan, root: Path) -> list[str]:
    lines = []
    for path in plan.unrecognized:
        rel = path.relative_to(root).as_posix()
        lines.append(f"unrecognized reference path, left in place: {rel}")
    if plan.ignored:
        names = ", ".join(p.relative_to(root).as_posix() for p in plan.ignored)
        lines.append(f"ignored {len(plan.ignored)} non-reference file(s): {names}")
    return lines


def _n_fixtures(n: int) -> str:
    return f"{n} fixture" if n == 1 else f"{n} fixtures"
