"""Derive trunk-U diagram specs from a live ModelConfig.

Torch-free: ModelConfig and the obs config it nests import only pure ints and
dataclasses, so building diagram rows pulls in none of the training stack's
torch weight. Param counts — which need an instantiated model — live elsewhere.

The trunk's encoder pushes one skip per ResBlock (including the strided level
transitions); the decoder pops them in reverse, so encoder block i pairs with
decoder block (last - i). `trunk_spec` reproduces that pairing as U rows.
"""

from __future__ import annotations

from dataclasses import dataclass

from model_arch.diagrams import HeadRow, NodeSpec, URow
from training.bc.model_config import ModelConfig


CONV = "ConvResBlock"
DECONV = "DeconvResBlock"


@dataclass(frozen=True)
class BlockFact:
    """The structured facts for one trunk ResBlock — what the diagram label is
    rendered from, and what the consistency test checks against the live model's
    conv layers. Hand-derived here from `ModelConfig`; `tests/` asserts it
    matches the instantiated `PyramidModule`."""

    name: str          # "outer_enc[0]", "strided_down_om", ...
    base: str          # CONV / DECONV
    kind: str          # "enc" / "dec" (box color)
    in_ch: int
    out_ch: int
    frm: int           # spatial side in
    to: int            # spatial side out
    stride: bool       # True iff a stride-2 (spatial-halving/doubling) block


@dataclass(frozen=True)
class TrunkSpec:
    granular: list[URow]      # one box per ResBlock
    compact: list[URow]       # per-level ResBlock runs collapsed to one "×N" box
    enc_facts: list[BlockFact]  # encoder blocks, push order (top → bottom of U)
    dec_facts: list[BlockFact]  # decoder blocks, execution order (inner → outer)
    in_label: str
    out_label: str
    n_skips: int


@dataclass(frozen=True)
class OverviewSpec:
    obs: NodeSpec
    trunk: NodeSpec
    embedding: NodeSpec
    heads: list[HeadRow]


def _chan(in_w: int, out_w: int) -> str:
    return f"{in_w} ch" if in_w == out_w else f"{in_w}→{out_w} ch"


def _spatial(frm: int, to: int) -> str:
    return f"{frm}²" if frm == to else f"{frm}²→{to}²"


def _detail(in_w: int, out_w: int, frm: int, to: int) -> str:
    return f"{_chan(in_w, out_w)} · {_spatial(frm, to)}"


def _node(f: BlockFact) -> NodeSpec:
    label = f"{f.base} · stride 2" if f.stride else f.base
    return NodeSpec(f.name, (label, _detail(f.in_ch, f.out_ch, f.frm, f.to)), f.kind)


def _agg(
    title: str,
    base: str,
    kind: str,
    count: int,
    w: int,
    spatial: int,
) -> NodeSpec | None:
    """Collapse a run of `count` identical ResBlocks into one box, or None if
    the level has no blocks."""
    if count == 0:
        return None
    detail = f"{w} ch · {spatial}²"
    if count == 1:
        return NodeSpec(title, (base, detail), kind)
    return NodeSpec(f"{title} ×{count}", (f"{count}× {base}", detail), kind)


def trunk_spec(cfg: ModelConfig) -> TrunkSpec:
    ow, mw, iw = cfg.outer_width, cfg.middle_width, cfg.inner_width
    n, mm, mi = cfg.n_outer, cfg.m_middle, cfg.m_inner
    # Spatial side length per contraction level (square board, halving twice).
    so, sm, si = cfg.H, cfg.H // 2, cfg.H // 4

    initial = NodeSpec(
        "initial_conv",
        ("3×3 conv + GN + ReLU", f"{cfg.in_ch}→{ow} ch · {so}²"),
        "aux",
    )

    # Strided level transitions — shared between the granular sequences and the
    # compact view (which keeps them explicit between aggregated level runs).
    sd_om = BlockFact("strided_down_om", CONV, "enc", ow, mw, so, sm, True)
    sd_mi = BlockFact("strided_down_mi", CONV, "enc", mw, iw, sm, si, True)
    su_im = BlockFact("strided_up_im", DECONV, "dec", iw, mw, si, sm, True)
    su_mo = BlockFact("strided_up_mo", DECONV, "dec", mw, ow, sm, so, True)

    # Encoder blocks in push order (top → bottom of the U).
    enc_facts = [
        *(BlockFact(f"outer_enc[{i}]", CONV, "enc", ow, ow, so, so, False)
          for i in range(n)),
        sd_om,
        *(BlockFact(f"middle_enc[{i}]", CONV, "enc", mw, mw, sm, sm, False)
          for i in range(mm)),
        sd_mi,
        *(BlockFact(f"inner_enc[{i}]", CONV, "enc", iw, iw, si, si, False)
          for i in range(mi)),
    ]
    # Decoder blocks in execution order (inner → outer); paired with encoder
    # blocks in reverse, mirroring the skip pop order.
    dec_facts = [
        *(BlockFact(f"inner_dec[{i}]", DECONV, "dec", iw, iw, si, si, False)
          for i in range(mi)),
        su_im,
        *(BlockFact(f"middle_dec[{i}]", DECONV, "dec", mw, mw, sm, sm, False)
          for i in range(mm)),
        su_mo,
        *(BlockFact(f"outer_dec[{i}]", DECONV, "dec", ow, ow, so, so, False)
          for i in range(n)),
    ]

    enc, dec = [_node(f) for f in enc_facts], [_node(f) for f in dec_facts]
    granular = [URow(initial, None)]
    for i, e in enumerate(enc):
        granular.append(URow(e, dec[-1 - i], skip=True))

    # Compact: aggregate the per-level runs, keep strided transitions explicit.
    levels = [
        (_agg("outer_enc", CONV, "enc", n, ow, so),
         _agg("outer_dec", DECONV, "dec", n, ow, so)),
        (_node(sd_om), _node(su_mo)),
        (_agg("middle_enc", CONV, "enc", mm, mw, sm),
         _agg("middle_dec", DECONV, "dec", mm, mw, sm)),
        (_node(sd_mi), _node(su_im)),
        (_agg("inner_enc", CONV, "enc", mi, iw, si),
         _agg("inner_dec", DECONV, "dec", mi, iw, si)),
    ]
    compact = [URow(initial, None)]
    for e, dc in levels:
        if e and dc:
            compact.append(URow(e, dc, skip=True))

    return TrunkSpec(
        granular=granular,
        compact=compact,
        enc_facts=enc_facts,
        dec_facts=dec_facts,
        in_label=f"obs · [B, {cfg.in_ch}, {cfg.H}, {cfg.W}]",
        out_label=f"trunk out · [B, {ow}, {cfg.H}, {cfg.W}]",
        n_skips=len(enc_facts),
    )


def overview_spec(cfg: ModelConfig) -> OverviewSpec:
    """Top-level figure spec. Unlike `trunk_spec`, the head boxes carry a hand-
    written gloss of each head's computation ("masked avg-pool", "4 dir × 2
    split") — semantic descriptions not derivable from the modules, and not
    guarded beyond the head-count check in tests/. Tensor shapes and Linear
    input dims track `cfg`; the rest is maintained by hand.
    """
    ow = cfg.outer_width
    hw = cfg.H * cfg.W
    obs = NodeSpec("obs", (f"[B, {cfg.in_ch}, {cfg.H}, {cfg.W}]",), "io")
    trunk = NodeSpec(
        "PyramidModule trunk",
        ("2-contraction U-Net",
         f"{cfg.outer_width}/{cfg.middle_width}/{cfg.inner_width} widths"),
        "aux",
    )
    embedding = NodeSpec("trunk embedding", (f"[B, {ow}, {cfg.H}, {cfg.W}]",), "io")

    policy = HeadRow(
        NodeSpec("PolicyHead", ("head PM (N=1) + 3×3 conv", "8 = 4 dir × 2 split"),
                 "head"),
        NodeSpec("policy_logits", (f"[B, 8, {cfg.H}, {cfg.W}]",), "io"),
    )
    pass_ = HeadRow(
        NodeSpec("PassHead", ("masked avg-pool", f"+ Linear({ow} → 1)"), "head"),
        NodeSpec("pass_logit", ("[B]",), "io"),
    )
    # ValueHead's readout tail is identical across variants; the `pyramid`
    # variant additionally runs an embedded head-PM over the trunk features
    # first, so the label must reflect the configured variant.
    v_pre = "head PM → " if cfg.value_head_variant == "pyramid" else ""
    value = HeadRow(
        NodeSpec(f"ValueHead ({cfg.value_head_variant})",
                 (f"{v_pre}conv → 1ch, mask, flatten", f"+ Linear({hw} → 8)"), "head"),
        NodeSpec("value_logits", ("[B, 8]",), "io"),
    )
    return OverviewSpec(obs, trunk, embedding, [policy, pass_, value])
