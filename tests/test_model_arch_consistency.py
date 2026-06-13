"""Guard the hand-mirrored architecture diagram against the live model.

`model_arch.spec` re-encodes the trunk's block structure and head set by hand —
it reads only `ModelConfig` (never an instantiated module) to stay torch-free,
so a topology change in `trunk.py` or a new head on `BCModel` won't error there;
it just silently emits a stale diagram. These tests instantiate the real model
and compare its conv layers / child modules against what the spec emits, turning
that silent drift into a loud failure.

The live model is walked *generically* (iterate children, flatten ModuleLists,
split on block class) rather than re-listing the level structure — otherwise the
test would be a third copy of the same hand-mirror and drift right along with it.
"""

from __future__ import annotations

from model_arch.spec import BlockFact, overview_spec, trunk_spec
import torch.nn as nn

from training.bc.model.bc_model import BCModel
from training.bc.model.trunk import ConvResBlock, DeconvResBlock
from training.bc.model_config import MODEL_CONFIG_DEFAULTS


def _live_blocks(trunk: nn.Module) -> tuple[list, list]:
    """(encoder, decoder) lists of (name, block) in execution order, read from
    the instantiated trunk. Child definition order is execution order;
    ConvResBlock is encoder-side, DeconvResBlock decoder-side."""
    enc, dec = [], []
    for name, child in trunk.named_children():
        members = list(child) if isinstance(child, nn.ModuleList) else [child]
        for j, m in enumerate(members):
            label = f"{name}[{j}]" if isinstance(child, nn.ModuleList) else name
            if isinstance(m, ConvResBlock):
                enc.append((label, m))
            elif isinstance(m, DeconvResBlock):
                dec.append((label, m))
    return enc, dec


def _assert_match(live: list, facts: list[BlockFact], in_conv: str, out_conv: str) -> None:
    assert len(live) == len(facts), (
        f"trunk block count drifted: live={len(live)} spec={len(facts)}"
    )
    for (name, blk), f in zip(live, facts, strict=True):
        assert f.name == name, f"block name drift: spec {f.name!r} vs live {name!r}"
        assert f.in_ch == getattr(blk, in_conv).in_channels, f"{name}: in_ch"
        assert f.out_ch == getattr(blk, out_conv).out_channels, f"{name}: out_ch"
        assert f.stride == (getattr(blk, in_conv).stride[0] == 2), f"{name}: stride"


def test_trunk_spec_matches_live_model() -> None:
    cfg = MODEL_CONFIG_DEFAULTS
    trunk = BCModel(cfg).trunk
    enc_live, dec_live = _live_blocks(trunk)
    spec = trunk_spec(cfg)

    # ConvResBlock: conv1 carries input channels + stride, conv2 the output.
    _assert_match(enc_live, spec.enc_facts, "conv1", "conv2")
    # DeconvResBlock: deconv1 carries input channels + stride, deconv2 the output.
    _assert_match(dec_live, spec.dec_facts, "deconv1", "deconv2")

    # One skip per encoder block, consumed by its mirror decoder block.
    assert spec.n_skips == len(enc_live) == len(dec_live)


def test_overview_head_count_matches_model() -> None:
    cfg = MODEL_CONFIG_DEFAULTS
    model = BCModel(cfg)
    head_children = [n for n, _ in model.named_children() if n != "trunk"]
    assert len(overview_spec(cfg).heads) == len(head_children)


def test_build_artifacts_nonempty() -> None:
    """Pins the regression where build_artifacts() returned an empty dict,
    silently making the regenerate/--check machinery a no-op."""
    import gen_model_arch_doc

    artifacts = gen_model_arch_doc.build_artifacts()
    assert artifacts
    assert all(content for content in artifacts.values())
