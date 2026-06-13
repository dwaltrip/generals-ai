"""Diagram builders: named figures composed from svgkit primitives.

Builders take plain-data specs (`NodeSpec` / `HeadRow` / `URow`) and lay out
their boxes and edges. They carry no torch and no model imports — `spec.py`
owns the translation from a live `ModelConfig` into these specs, keeping the
drawing code free of model knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import drawsvg as dw

from model_arch.svgkit import BOX_W, Diagram, Point, box_height
from model_arch.theme import PALETTE


@dataclass(frozen=True)
class NodeSpec:
    title: str
    subs: tuple[str, ...] = ()
    kind: str = "aux"


@dataclass(frozen=True)
class HeadRow:
    """One output head and the tensor it produces, drawn as a fan-out branch
    in the top-level overview."""

    head: NodeSpec
    output: NodeSpec


@dataclass(frozen=True)
class URow:
    """One row of a U diagram: encoder-side box, decoder-side box, and
    whether a skip connection bridges them. Either side may be None
    (e.g. a stem row with no decoder mirror)."""

    enc: NodeSpec | None
    dec: NodeSpec | None
    skip: bool = False


COL_SPAN = 180   # horizontal gap between the two columns (skip-line length)
ROW_GAP = 28
U_DROP = 40      # how far the bottom elbow dips below the last row
IO_STUB = 26     # length of the input/output arrow stub above/below a column
LABEL_PAD = 8    # gap between a stub's far end and its text label
NOTE_GAP = 22    # gap below the U elbow to the skip-legend note


def u_diagram(
    rows: list[URow],
    in_label: str,
    out_label: str,
    skip_note: str | None = None,
) -> dw.Drawing:
    """Two-column U: encoder flows down the left column, decoder flows up
    the right column, joined by an elbow under the bottom row. Skips are
    horizontal dashed arrows between mirror-paired rows."""
    d = Diagram(PALETTE)
    enc_x = 0
    dec_x = BOX_W + COL_SPAN

    # Place boxes row by row; both sides of a row share a top y.
    enc_boxes, dec_boxes = [], []
    y = 0.0
    for row in rows:
        placed_h = 0.0
        if row.enc:
            b = d.box(enc_x, y, row.enc.title, row.enc.subs, row.enc.kind)
            enc_boxes.append(b)
            placed_h = max(placed_h, b.h)
        if row.dec:
            b = d.box(dec_x, y, row.dec.title, row.dec.subs, row.dec.kind)
            dec_boxes.append(b)
            placed_h = max(placed_h, b.h)
        if row.skip:
            assert row.enc and row.dec, "skip row needs boxes on both sides"
            d.skip(enc_boxes[-1].right, dec_boxes[-1].left)
        y += placed_h + ROW_GAP

    # Column flows: down the encoder, up the decoder.
    for a, b in pairwise(enc_boxes):
        d.flow(a.bottom, b.top)
    for upper, lower in pairwise(dec_boxes):
        d.flow(lower.top, upper.bottom)

    # Bottom elbow: last encoder box → under the U → up into last decoder box.
    e_last, d_last = enc_boxes[-1], dec_boxes[-1]
    dip_y = max(e_last.bottom.y, d_last.bottom.y) + U_DROP
    mid_x = (e_last.bottom.x + d_last.bottom.x) / 2
    d.elbow(e_last.bottom, Point(e_last.bottom.x, dip_y),
            Point(d_last.bottom.x, dip_y), d_last.bottom)

    # Input / output arrows + labels above the columns. The label sits one
    # LABEL_PAD beyond the stub's far end.
    e_first, d_first = enc_boxes[0], dec_boxes[0]
    d.note(e_first.top.x, e_first.top.y - IO_STUB - LABEL_PAD, in_label)
    d.flow(Point(e_first.top.x, e_first.top.y - IO_STUB), e_first.top)
    d.note(d_first.top.x, d_first.top.y - IO_STUB - LABEL_PAD, out_label)
    d.flow(d_first.top, Point(d_first.top.x, d_first.top.y - IO_STUB))

    # Skip-connection legend: open space just below the U elbow.
    if skip_note:
        d.note(mid_x, dip_y + NOTE_GAP, skip_note, cls="note note-skip")

    return d.render()


# --- top-level overview ---

OV_COL_GAP = 60   # horizontal gap between overview columns
OV_HEAD_GAP = 26  # vertical gap between stacked heads


def overview_diagram(
    obs: NodeSpec,
    trunk: NodeSpec,
    embedding: NodeSpec,
    heads: list[HeadRow],
) -> dw.Drawing:
    """Left-to-right spine (obs → trunk → embedding) that fans out to the
    output heads, each producing its own tensor. The spine is centered on the
    head stack; fan-out branches route through a shared vertical bus."""
    d = Diagram(PALETTE)
    x_obs = 0.0
    x_trunk = x_obs + BOX_W + OV_COL_GAP
    x_emb = x_trunk + BOX_W + OV_COL_GAP
    x_head = x_emb + BOX_W + OV_COL_GAP
    x_out = x_head + BOX_W + OV_COL_GAP

    # Heads + their output tensors, stacked top-to-bottom (tops aligned).
    head_boxes, out_boxes = [], []
    y = 0.0
    for hr in heads:
        hb = d.box(x_head, y, hr.head.title, hr.head.subs, hr.head.kind)
        ob = d.box(x_out, y, hr.output.title, hr.output.subs, hr.output.kind)
        head_boxes.append(hb)
        out_boxes.append(ob)
        y += max(hb.h, ob.h) + OV_HEAD_GAP

    # Spine boxes vertically centered on the head stack.
    mid_y = (head_boxes[0].y + head_boxes[-1].y + head_boxes[-1].h) / 2

    def centered(x: float, node: NodeSpec):
        h = box_height(len(node.subs))
        return d.box(x, mid_y - h / 2, node.title, node.subs, node.kind)

    obs_b = centered(x_obs, obs)
    trunk_b = centered(x_trunk, trunk)
    emb_b = centered(x_emb, embedding)

    d.flow(obs_b.right, trunk_b.left)
    d.flow(trunk_b.right, emb_b.left)

    # Fan-out: embedding → each head. Co-linear branch goes straight; others
    # route through a shared vertical bus at branch_x.
    branch_x = (emb_b.right.x + x_head) / 2
    for hb in head_boxes:
        if abs(hb.left.y - emb_b.right.y) < 0.5:
            d.flow(emb_b.right, hb.left)
        else:
            d.elbow(emb_b.right, Point(branch_x, emb_b.right.y),
                    Point(branch_x, hb.left.y), hb.left)

    for hb, ob in zip(head_boxes, out_boxes, strict=True):
        d.flow(hb.right, ob.left)

    return d.render()
