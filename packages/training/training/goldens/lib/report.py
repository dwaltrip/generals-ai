"""The regen report: a text summary of a RegenPlan."""

from __future__ import annotations

from dataclasses import dataclass

from training.goldens.lib.content import EntryContent, content_id
from training.goldens.lib.diff import ArrayChange
from training.goldens.lib.entry import Changed, Moved, New, PlannedEntry, Unchanged
from training.goldens.lib.plan import RegenPlan, SurfacePlan, count_statuses
from training.goldens.lib.render import render_change


def render_regen_report(plan: RegenPlan) -> str:
    numbers = {f.id: i + 1 for i, f in enumerate(plan.fixtures)}
    lines = ["regen: compared against the references on disk before this run", ""]
    lines.append(_render_headline(_headline_counts(plan)))

    for surface in plan.surfaces:
        body = _render_entry_groups(_entry_groups(surface), numbers)
        if not surface.planned_entries:
            body.append("  warning: no points registered")
        for section in _array_sections(surface, list(numbers)):
            body += _render_array_section(section, numbers)
        if body:
            lines += ["", surface.surface, *body]

    stray = _render_stray_files(plan)
    if stray:
        lines += ["", *stray]
    return "\n".join(lines) + "\n"


# --- Data transforms ---


@dataclass(frozen=True)
class _HeadlineCounts:
    changed: int
    new: int
    moved: int
    removed: int
    unchanged: int


# New, moved, or removed entries that share a point (and a move source), with
# their fixtures.
@dataclass(frozen=True)
class _EntryGroup:
    kind: str                   # "new", "moved", or "removed"
    point: str
    source: str | None          # for a move, the point it came from
    fixtures: tuple[str, ...]


# Points with the same change to an array on one fixture, and the same content
# before and after. The labels index the distinct contents on that fixture, and
# are None where the array is absent.
@dataclass(frozen=True)
class _PointLine:
    points: tuple[str, ...]
    change: ArrayChange | None   # None: unchanged
    before: int | None
    after: int | None


@dataclass(frozen=True)
class _FixtureBlock:
    fixture: str
    lines: tuple[_PointLine, ...]


# One array's changes on one surface. Fixtures where the array changed on no
# point are only counted.
@dataclass(frozen=True)
class _ArraySection:
    name: str
    blocks: tuple[_FixtureBlock, ...]
    n_unchanged_fixtures: int


def _headline_counts(plan: RegenPlan) -> _HeadlineCounts:
    counts = count_statuses(plan.planned_entries())
    return _HeadlineCounts(
        changed=counts[Changed],
        new=counts[New],
        moved=counts[Moved],
        removed=sum(len(s.planned_removals) for s in plan.surfaces),
        unchanged=counts[Unchanged],
    )


def _entry_groups(surface: SurfacePlan) -> list[_EntryGroup]:
    keyed: list[tuple[str, str, str | None, str]] = []
    for e in surface.planned_entries:
        match e.status:
            case New():
                keyed.append(("new", e.id.point, None, e.id.fixture))
            case Moved(source=source):
                keyed.append(("moved", e.id.point, source.point, e.id.fixture))
            case _:
                pass
    keyed += [("removed", eid.point, None, eid.fixture) for eid in surface.planned_removals]

    groups: dict[tuple[str, str, str | None], list[str]] = {}
    for kind, point, source, fixture in keyed:
        groups.setdefault((kind, point, source), []).append(fixture)
    return [
        _EntryGroup(kind=kind, point=point, source=source, fixtures=tuple(fixtures))
        for (kind, point, source), fixtures in groups.items()
    ]


def _array_sections(surface: SurfacePlan, fixtures: list[str]) -> list[_ArraySection]:
    compared = [e for e in surface.planned_entries if isinstance(e.status, Changed | Unchanged)]
    names = sorted({
        c.name for e in compared if isinstance(e.status, Changed) for c in e.status.diff.changes
    })
    return [_array_section(name, compared, fixtures) for name in names]


def _array_section(name: str, compared: list[PlannedEntry], fixtures: list[str]) -> _ArraySection:
    blocks = []
    n_unchanged = 0
    for fixture in fixtures:
        on_fixture = [
            e for e in compared
            if e.id.fixture == fixture and (name in e.content or name in _stored(e))
        ]
        if not on_fixture:
            continue
        if all(_change_to(e, name) is None for e in on_fixture):
            n_unchanged += 1
        else:
            blocks.append(_FixtureBlock(fixture=fixture, lines=_point_lines(name, on_fixture)))
    return _ArraySection(name=name, blocks=tuple(blocks), n_unchanged_fixtures=n_unchanged)


# The entries are on one fixture, in registry order.
def _point_lines(name: str, entries: list[PlannedEntry]) -> tuple[_PointLine, ...]:
    labels: dict[str, int] = {}

    def label(content: EntryContent) -> int | None:
        if name not in content:
            return None
        return labels.setdefault(content_id(content[name]), len(labels))

    grouped: dict[tuple[ArrayChange | None, int | None, int | None], list[str]] = {}
    for e in entries:
        key = (_change_to(e, name), label(_stored(e)), label(e.content))
        grouped.setdefault(key, []).append(e.id.point)
    return tuple(
        _PointLine(points=tuple(points), change=change, before=before, after=after)
        for (change, before, after), points in grouped.items()
    )


# The content before the run, for an entry that was compared.
def _stored(e: PlannedEntry) -> EntryContent:
    return e.status.stored if isinstance(e.status, Changed) else e.content


def _change_to(e: PlannedEntry, name: str) -> ArrayChange | None:
    if not isinstance(e.status, Changed):
        return None
    return next((c for c in e.status.diff.changes if c.name == name), None)


# --- Formatting helpers ---


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# A, B, ..., Z, AA, AB, ...
def _letters(i: int) -> str:
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = chr(ord("A") + r) + out
    return out


def _content_label(fixture_number: int, label: int | None) -> str:
    return "" if label is None else f"{fixture_number}-{_letters(label)}"


def _fixture_heading(fixture: str, numbers: dict[str, int]) -> str:
    return f"Fixture {numbers[fixture]} ({fixture})"


# Fixtures that aren't registered (a removed fixture's files) are shown by id.
def _fixture_set(fixtures: tuple[str, ...], numbers: dict[str, int]) -> str:
    if set(fixtures) == set(numbers):
        return _plural(len(fixtures), "fixture")
    names = [f"{numbers[f]} ({f})" if f in numbers else f for f in fixtures]
    return f"{'fixture' if len(names) == 1 else 'fixtures'} {', '.join(names)}"


# The width of each column but the last, for aligning rows.
def _column_widths(rows: list[tuple[str, ...]]) -> list[int]:
    if not rows:
        return []
    return [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]


def _format_row(row: tuple[str, ...], widths: list[int], indent: str) -> str:
    cells = [cell.ljust(w) for cell, w in zip(row[:-1], widths, strict=True)] + [row[-1]]
    return (indent + "   ".join(cells)).rstrip()


# --- Rendering ---


def _render_headline(c: _HeadlineCounts) -> str:
    if not (c.changed or c.new or c.moved or c.removed):
        return f"no changes   unchanged {c.unchanged}"
    return (
        f"changed {c.changed}   new {c.new}   moved {c.moved}"
        f"   removed {c.removed}   unchanged {c.unchanged}"
    )


def _render_entry_groups(groups: list[_EntryGroup], numbers: dict[str, int]) -> list[str]:
    rows = []
    for g in groups:
        point = g.point if g.source is None else f"{g.point} <- {g.source}"
        rows.append((g.kind, point, _fixture_set(g.fixtures, numbers)))
    widths = _column_widths(rows)
    return [_format_row(row, widths, indent="  ") for row in rows]


def _render_array_section(section: _ArraySection, numbers: dict[str, int]) -> list[str]:
    block_rows = [
        [_point_row(line, numbers[block.fixture]) for line in block.lines]
        for block in section.blocks
    ]
    # Rows are aligned across all of the section's fixture blocks.
    widths = _column_widths([row for rows in block_rows for row in rows])

    lines = [f"  {section.name}"]
    for block, rows in zip(section.blocks, block_rows, strict=True):
        lines.append(f"    {_fixture_heading(block.fixture, numbers)}")
        lines += [_format_row(row, widths, indent="      ") for row in rows]
    if section.n_unchanged_fixtures:
        lines.append(f"    unchanged on {_plural(section.n_unchanged_fixtures, 'fixture')}")
    return lines


def _point_row(line: _PointLine, fixture_number: int) -> tuple[str, str, str]:
    before = _content_label(fixture_number, line.before)
    after = _content_label(fixture_number, line.after)
    if line.change is None:
        return (", ".join(line.points), "unchanged", before)
    return (", ".join(line.points), render_change(line.change), f"{before} -> {after}".strip())


def _render_stray_files(plan: RegenPlan) -> list[str]:
    lines = [f"unrecognized reference path, left in place: {p}" for p in plan.tree.unrecognized]
    if plan.tree.ignored:
        names = ", ".join(str(p) for p in plan.tree.ignored)
        lines.append(f"ignored {_plural(len(plan.tree.ignored), 'non-reference file')}: {names}")
    return lines
