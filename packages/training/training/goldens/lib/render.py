from __future__ import annotations

from training.goldens.lib.content import ArrayLayout
from training.goldens.lib.diff import (
    ArrayAdded,
    ArrayChange,
    ArrayLayoutChanged,
    ArrayRemoved,
    ArrayRowsChanged,
    ContentDiff,
)


def render_rows(rows: tuple[int, ...], n: int) -> str:
    assert rows, "no rows"
    first, last = rows[0], rows[-1]
    if len(rows) == 1:
        return f"row {first} of {n}"
    if last - first + 1 == len(rows):
        return f"rows {first}..{last} of {n}"
    return f"{len(rows)} of {n} rows, from {first}"


def render_layout(layout: ArrayLayout) -> str:
    out = f"{layout.dtype} {layout.shape}"
    if layout.hashed_axis is not None:
        out += f", hashed on axis {layout.hashed_axis}"
    return out


# The array's name is left out, since the report shows it in a heading.
def render_change(change: ArrayChange) -> str:
    match change:
        case ArrayAdded(got=got):
            return f"added, {render_layout(got)}"
        case ArrayRemoved(stored=stored):
            return f"removed, {render_layout(stored)}"
        case ArrayLayoutChanged(stored=stored, got=got):
            return f"layout {render_layout(stored)} -> {render_layout(got)}"
        case ArrayRowsChanged(layout=layout, rows=rows):
            n = layout.shape[layout.hashed_axis or 0]
            return render_rows(rows, n)


def render_content_diff(diff: ContentDiff) -> str:
    return "\n".join(f"{c.name}: {render_change(c)}" for c in diff.changes)
