"""Second-level SVG primitives for architecture diagrams, on top of drawsvg.

Model-agnostic: this module knows boxes, anchors, arrows, and stacks — never
encoders or heads. Diagram builders (`diagrams.py`) compose these.

Conventions:
  - Coordinates are absolute SVG user units; layout is plain arithmetic done
    by callers. No layout engine.
  - Visual style lives in `CSS`. Each box gets a `kind` string ("enc", "dec",
    "aux") that maps to CSS classes; geometry never encodes style.
  - Arrows attach to box edge anchors (`box.top`, `.right`, ...) and stop
    `ARROW_GAP` short of them, with the marker tip at the line end, so
    arrowheads never overlap their target box.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import drawsvg as dw


# Box text layout (px). A box is a title row plus zero or more subtitle rows;
# height follows from the row count.
BOX_W = 200
TITLE_CY = 17        # title center, from box top
SUB_DY = 16          # spacing between text rows
BOX_PAD_BOTTOM = 15

ARROW_GAP = 2.5      # gap between arrow tip and the target box edge

_FONT = '-apple-system, "Segoe UI", "Helvetica Neue", sans-serif'

# Structural CSS: box/edge/annotation styling that holds for any diagram. No
# box-`kind` colors live here — those are the one diagram-semantic piece and
# come from PALETTE below. `.skip` is a generic secondary (dashed) edge; the
# diagram layer decides what it means.
STRUCTURAL_CSS = f"""
.node rect {{
  stroke-width: 0.5;
  rx: 8;
}}
.node text {{
  text-anchor: middle;
  dominant-baseline: central;
  font-family: {_FONT};
}}
.node .bt {{
  font-size: 14px;
  font-weight: 500;
}}
.node .bs {{
  font-size: 12px;
}}
.flow {{
  stroke: #73726c;
  stroke-width: 1.5;
  fill: none;
}}
.skip {{
  stroke: #d85a30;
  stroke-width: 1.5;
  fill: none;
  stroke-dasharray: 6 5;
}}
.note {{
  font-size: 12px;
  fill: #3d3d3a;
  text-anchor: middle;
  font-family: {_FONT};
}}
.note-skip {{
  fill: #993c1d;
}}
"""


@dataclass(frozen=True)
class KindStyle:
    """Colors for one box `kind`: fill/stroke style the rect, title/sub the
    two text rows."""

    fill: str
    stroke: str
    title: str
    sub: str


def palette_css(palette: dict[str, KindStyle]) -> str:
    """Render a `kind -> KindStyle` palette into CSS rules.

    svgkit ships no palette of its own — a box just emits `class="node <kind>"`
    and the diagram layer supplies which kinds exist and their colors (see
    `theme.py`). Insertion order is the emit order; keep palettes ordered so the
    generated SVG is deterministic.
    """
    blocks = []
    for kind, s in palette.items():
        blocks.append(
            f".{kind} rect {{\n"
            f"  fill: {s.fill};\n"
            f"  stroke: {s.stroke};\n"
            f"}}\n"
            f".{kind} .bt {{\n"
            f"  fill: {s.title};\n"
            f"}}\n"
            f".{kind} .bs {{\n"
            f"  fill: {s.sub};\n"
            f"}}"
        )
    return "\n".join(blocks)


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Box:
    """A placed node: geometry + label content. Anchors are edge midpoints."""

    x: float
    y: float
    w: float
    h: float
    title: str
    subs: tuple[str, ...]
    kind: str

    @property
    def top(self) -> Point:
        return Point(self.x + self.w / 2, self.y)

    @property
    def bottom(self) -> Point:
        return Point(self.x + self.w / 2, self.y + self.h)

    @property
    def left(self) -> Point:
        return Point(self.x, self.y + self.h / 2)

    @property
    def right(self) -> Point:
        return Point(self.x + self.w, self.y + self.h / 2)

    @property
    def center(self) -> Point:
        return Point(self.x + self.w / 2, self.y + self.h / 2)


def box_height(n_subs: int) -> float:
    if n_subs == 0:
        return 32
    return TITLE_CY + SUB_DY * n_subs + BOX_PAD_BOTTOM + 1


def _shorten(a: Point, b: Point, d: float) -> Point:
    """Move `b` toward `a` by distance `d` (straight segments only)."""
    dx, dy = b.x - a.x, b.y - a.y
    length = (dx * dx + dy * dy) ** 0.5
    if length <= d:
        return b
    f = (length - d) / length
    return Point(a.x + dx * f, a.y + dy * f)


class Diagram:
    """Accumulates elements + tracks extents so the canvas can auto-size.

    Callers place boxes at absolute coordinates, connect them with
    `flow` / `elbow` / `skip`, then call `render()`. The `palette` supplies
    colors for whatever box `kind`s the caller uses.
    """

    def __init__(self, palette: dict[str, KindStyle]) -> None:
        self._palette = palette
        self._elements: list = []
        # Extent tracking: every drawn element registers the points it spans.
        self._xs: list[float] = []
        self._ys: list[float] = []
        # scale is float at runtime; drawsvg ships no py.typed so pyright infers
        # the param as int from its default. Targeted ignore beats disabling
        # reportArgumentType across all of tools/.
        marker = dw.Marker(
            -0.1, -5, 11, 5,
            scale=0.6,  # pyright: ignore[reportArgumentType]
            orient="auto-start-reverse",
            refX=8,
        )
        marker.append(dw.Lines(
            2, -4, 8, 0, 2, 4,
            fill="none",
            stroke="context-stroke",
            stroke_width=1.5,
            stroke_linecap="round",
        ))
        self._arrow = marker

    def _track(self, *pts: Point) -> None:
        for p in pts:
            self._xs.append(p.x)
            self._ys.append(p.y)

    # --- nodes ---

    def box(self, x: float, y: float, title: str, subs: Sequence[str] = (),
            kind: str = "aux", w: float = BOX_W) -> Box:
        b = Box(x, y, w, box_height(len(subs)), title, tuple(subs), kind)
        g = dw.Group(class_=f"node {kind}")
        g.append(dw.Rectangle(b.x, b.y, b.w, b.h))
        cx = b.center.x
        if b.subs:
            g.append(dw.Text(b.title, 14, cx, b.y + TITLE_CY, class_="bt"))
            for i, sub in enumerate(b.subs):
                g.append(dw.Text(sub, 12, cx, b.y + TITLE_CY + SUB_DY * (i + 1),
                                 class_="bs"))
        else:
            g.append(dw.Text(b.title, 14, cx, b.center.y, class_="bt"))
        self._elements.append(g)
        self._track(Point(b.x, b.y), Point(b.x + b.w, b.y + b.h))
        return b

    # --- edges ---

    def flow(self, a: Point, b: Point, cls: str = "flow") -> None:
        """Straight arrow a → b, tip stopping ARROW_GAP short of b."""
        end = _shorten(a, b, ARROW_GAP)
        self._elements.append(dw.Line(a.x, a.y, end.x, end.y, class_=cls,
                                      marker_end=self._arrow))
        self._track(a, b)

    def skip(self, a: Point, b: Point) -> None:
        self.flow(a, b, cls="skip")

    def elbow(self, *pts: Point, cls: str = "flow") -> None:
        """Axis-aligned polyline arrow through `pts`, arrowhead at the last."""
        assert len(pts) >= 2
        end = _shorten(pts[-2], pts[-1], ARROW_GAP)
        d = f"M {pts[0].x} {pts[0].y}"
        for p in pts[1:-1]:
            d += f" L {p.x} {p.y}"
        d += f" L {end.x} {end.y}"
        self._elements.append(dw.Path(d=d, class_=cls, marker_end=self._arrow))
        self._track(*pts)

    # --- annotations ---

    def note(self, x: float, y: float, text: str, cls: str = "note") -> None:
        self._elements.append(dw.Text(text, 12, x, y, class_=cls))
        # Rough text extent so render() leaves room: ~6.5px/char at 12px.
        half = len(text) * 6.5 / 2
        self._track(Point(x - half, y - 12), Point(x + half, y + 4))

    # --- output ---

    def render(self, pad: float = 24) -> dw.Drawing:
        x0, x1 = min(self._xs) - pad, max(self._xs) + pad
        y0, y1 = min(self._ys) - pad, max(self._ys) + pad
        w, h = x1 - x0, y1 - y0
        d = dw.Drawing(w, h, origin=(x0, y0))
        d.append_css(STRUCTURAL_CSS + palette_css(self._palette))
        for el in self._elements:
            d.append(el)
        return d
