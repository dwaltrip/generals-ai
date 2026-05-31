"""
Behavioral-cloning model: trunk (Pyramid Module) + policy / pass / value heads.

The trunk is a 2-contraction U-Net "Pyramid Module" — the inherited DeepNash
architecture (Perolat 2022, arXiv:2206.15378, supplementary Fig. 7) at the
half-width 128/128/160 variant. The shape:

    32² → 16² → 8² → 16² → 32²    (contract-bottleneck-expand)

Three heads read off the trunk's [B, C, H, W] spatial embedding:
  - policy: per-cell directional logits in NCHW [B, 8, H, W]
  - pass:   single scalar logit per sample (split-head pass design)
  - value:  categorical placement logits [B, 8]

The model returns a dict so `bc/loss.py` can address each head's output
independently and the overfit harness can log component-wise losses.

Two deviations from the inherited design (`network-architecture-design.md`)
are flagged in code:
  1. GroupNorm in place of LayerNorm — see `_gn` for the reasoning.
  2. Inline value head (option B from 5.20-2 session) — flagged in
     `ValueHead`. Extension path back to the full-spec DeepNash head
     is marked at the relevant `TODO(option C extension)` sites.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bc.model_config import VALUE_HEAD_VARIANTS, ModelConfig


# GroupNorm preferred group count. The trunk uses {64, 80, 128, 160}-channel
# tensors; 80 doesn't divide by 32, so `_gn` falls to 16 there. See `_gn`.
GN_GROUPS_PREFERRED = 32


def _gn(channels: int) -> nn.GroupNorm:
    """
    GroupNorm helper. Used everywhere a normalization layer is needed in
    this model.

    Why GroupNorm rather than LayerNorm (which the design doc tentatively
    picked) or BatchNorm (the CNN default)?

      - BatchNorm normalizes across the batch dim. In RL / policy-gradient
        training, batches are highly correlated (same trajectory, same
        perspective) and BN's running statistics drift in pathological ways.
        This is a well-known failure mode in RL — DeepNash, R-NaD, and most
        policy-gradient implementations avoid BN.
      - LayerNorm normalizes per-sample over the feature dim. Safe for RL,
        but `nn.LayerNorm` in PyTorch expects normalization over the *last*
        dim(s) of an input — awkward on NCHW conv tensors (need a permute
        dance or a custom op).
      - GroupNorm normalizes per-sample over channel groups (within a
        sample). Same intent as LayerNorm — independent of batch — but
        operates natively on NCHW conv tensors. The standard "LN-in-spirit
        but for ConvNets" choice (Wu & He 2018).

    32 groups is the GroupNorm-paper default for vision and what we use
    when possible. The trunk's middle→inner strided ResBlock has an
    80-channel C/2 intermediate (160/2 = 80) which doesn't divide by 32 —
    falls to 16 groups there. All other trunk channels accept 32.
    """
    for g in (GN_GROUPS_PREFERRED, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    raise ValueError(f"unreachable: channels={channels}")


# ---------------------------------------------------------------------------
# Residual blocks
# ---------------------------------------------------------------------------


class ConvResBlock(nn.Module):
    """
    DeepNash-spec convolutional residual block (encoder side).

    Structure (Perolat 2022 supplementary Fig. 7):

        input x ─┬─→ skip-out (exposed for the symmetric DeconvResBlock)
                 │
                 ├─→ conv(C/2, kernel=3, stride=S) → GN → ReLU
                 │   conv(C,   kernel=3, stride=1) → GN → ReLU
                 │                                          │
                 └─→ residual proj (1×1 conv if stride>1) ──+─→ output

    The `C → C/2 → C` shape is the standard "bottleneck" residual block
    (He et al. 2016): squeeze channels in the middle, restore at the end.
    Saves compute over a `C → C → C` block at the cost of representational
    capacity at the squeezed mid-layer.

    The skip-out is taken at the block's *input*, not its output —
    confirmed against the DeepNash paper. Each ConvResBlock contributes
    one skip to the symmetric DeconvResBlock on the decoder side.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        mid_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=stride, padding=1)
        self.gn1 = _gn(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.gn2 = _gn(out_ch)

        # Residual projection: identity unless we need a shape change.
        # Either striding (spatial halves) or channel widening triggers
        # the 1×1 conv. The strided level boundaries (outer→middle and
        # middle→inner) both end up here.
        if stride > 1 or in_ch != out_ch:
            self.residual_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns `(output, skip_out)`. The caller (PyramidModule) is
        responsible for stashing `skip_out` and routing it to the mirror
        decoder block.
        """
        skip_out = x  # taken at INPUT per the DeepNash spec
        y = self.conv1(x)
        y = self.gn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.gn2(y)
        y = F.relu(y)
        y = y + self.residual_proj(x)
        return y, skip_out


class DeconvResBlock(nn.Module):
    """
    DeepNash-spec deconvolutional residual block (decoder side).

    Structure (Perolat 2022 supplementary Fig. 7):

        input x ─→ deconvT(C/2, kernel=3, stride=S) → GN → ReLU
                                                            │
                   skip_in ── 1×1 deconvT (no activation) ──+
                                                            │
                   deconvT(C, kernel=3, stride=1) → GN → ReLU
                                                            │
                   residual proj (1×1 deconvT if stride>1) ─+─→ output

    Two things in here are non-standard for a U-Net:
      1. The skip is *summed* (not concatenated) into the C/2 intermediate,
         between the two deconvs. Most U-Nets concat the skip at the block
         input. The DeepNash variant keeps channel counts smaller — no
         doubling from concat — at the cost of less expressiveness.
      2. The skip is routed through its own 1×1 transposed conv (no
         activation) to map source channels → C/2. By construction, the
         skip source's spatial dim equals the post-first-deconv spatial
         dim, so the projection's stride is always 1 — pure channel
         reduction. (The DeepNash paper's "optional S" on this projection
         is a no-op in our wiring; see 5.20-2 session note.)

    "deconvT" here is `nn.ConvTranspose2d`. At stride=1 it's functionally
    a `nn.Conv2d`, but the DeepNash spec uses transposed-conv idiom
    throughout the decoder; matching keeps the lineage explicit.
    """

    def __init__(self, in_ch: int, out_ch: int, skip_in_ch: int, stride: int = 1):
        super().__init__()
        mid_ch = out_ch // 2

        # First deconv: stride S to upsample (when stride > 1).
        # output_padding = stride - 1 makes a stride-S kernel-3 padding-1
        # conv-transpose double the spatial dim exactly (for even input).
        self.deconv1 = nn.ConvTranspose2d(
            in_ch, mid_ch, kernel_size=3, stride=stride,
            padding=1, output_padding=(stride - 1),
        )
        self.gn1 = _gn(mid_ch)

        # Skip projection: 1×1 transposed conv, no activation. Spatial dim
        # of the skip source always matches the post-first-deconv spatial,
        # so stride=1 here; pure channel reduction `skip_in_ch → C/2`.
        self.skip_proj = nn.ConvTranspose2d(
            skip_in_ch, mid_ch, kernel_size=1, stride=1, padding=0,
        )

        # Second deconv: no stride, restores channels.
        self.deconv2 = nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.gn2 = _gn(out_ch)

        # Residual projection: same condition as the ConvResBlock's.
        # Separate projection from `skip_proj` — they live on different
        # paths through the block. The DeepNash Fig. 7 caption is explicit
        # about this ("residual connections are also processed by a
        # convolution layer with 1×1 kernel (hidden for clarity)").
        if stride > 1 or in_ch != out_ch:
            self.residual_proj = nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=1, stride=stride,
                output_padding=(stride - 1),
            )
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor, skip_in: torch.Tensor) -> torch.Tensor:
        y = self.deconv1(x)
        y = self.gn1(y)
        y = F.relu(y)
        # Skip summed in the middle of the block, at the C/2 intermediate.
        y = y + self.skip_proj(skip_in)
        y = self.deconv2(y)
        y = self.gn2(y)
        y = F.relu(y)
        y = y + self.residual_proj(x)
        return y


# ---------------------------------------------------------------------------
# Pyramid Module (U-Net torso)
# ---------------------------------------------------------------------------


class PyramidModule(nn.Module):
    """
    2-contraction U-Net pyramid per the DeepNash spec.

    Encoder path (spatial halving via strided ResBlocks):

        initial conv → outer ResBlocks → strided down → middle ResBlocks
                     → strided down → inner ResBlocks (bottleneck)

    Decoder path (spatial doubling via strided deconv ResBlocks, mirror):

        inner deconv ResBlocks → strided up → middle deconv ResBlocks
                               → strided up → outer deconv ResBlocks

    Skip topology: per-block, mirror-paired. Every ConvResBlock (including
    the strided ones) exposes its input as a skip; the symmetric
    DeconvResBlock consumes it via the 1×1 projection-and-sum described
    in DeconvResBlock's docstring. With (N, M, M) blocks per stack, that's
    `N + 1 + M + 1 + M = N + 2M + 2` skips total.

    Channel widening between levels (when widths differ across levels) is
    absorbed *inside* the strided ResBlock: its `out_ch` is the next
    level's width, its C/2 intermediate is `out_ch/2`, and the residual
    1×1 projection handles the in→out channel change in lockstep with the
    spatial halving.

    Used in three places:
      - The main trunk: N=2 / M=2 / M=2 at half-widths 128/128/160.
      - The policy head: N=1 / M=0 / M=0 at uniform width (the "head
        Pyramid Module" pattern from DeepNash §4).
      - (Optional, deferred) The value head, if we extend to option C
        from the 5.20-2 session note.

    `widths` is (outer, middle, inner) channel counts.
    """

    def __init__(
        self,
        in_ch: int,
        n_outer: int,
        m_middle: int,
        m_inner: int,
        widths: tuple[int, int, int],
    ):
        super().__init__()
        outer_w, middle_w, inner_w = widths

        # Initial 3×3 conv: projects in_ch into the outer width. This is
        # step 1 of the DeepNash Pyramid Module spec — it lifts the input
        # from observation channels into the trunk's working channel space.
        self.initial_conv = nn.Conv2d(in_ch, outer_w, kernel_size=3, padding=1)
        self.initial_gn = _gn(outer_w)

        # --- Encoder ---
        self.outer_enc = nn.ModuleList(
            [ConvResBlock(outer_w, outer_w, stride=1) for _ in range(n_outer)]
        )
        # Outer → middle: spatial halves; channel width changes if
        # outer_w != middle_w (no-op when they're equal, like the
        # 128/128/160 v1 config at this boundary).
        self.strided_down_om = ConvResBlock(outer_w, middle_w, stride=2)

        self.middle_enc = nn.ModuleList(
            [ConvResBlock(middle_w, middle_w, stride=1) for _ in range(m_middle)]
        )
        # Middle → inner: spatial halves; widens 128 → 160 at the v1 config.
        self.strided_down_mi = ConvResBlock(middle_w, inner_w, stride=2)

        self.inner_enc = nn.ModuleList(
            [ConvResBlock(inner_w, inner_w, stride=1) for _ in range(m_inner)]
        )

        # --- Decoder (mirror) ---
        # Inner-level deconv ResBlocks pair with inner_enc[::-1].
        # Each skip-in has channels = inner_w (the encoder block's input width).
        self.inner_dec = nn.ModuleList(
            [DeconvResBlock(inner_w, inner_w, skip_in_ch=inner_w, stride=1)
             for _ in range(m_inner)]
        )
        # Inner → middle (strided up). Pairs with strided_down_mi, whose
        # input was at middle_w spatial 16² → so skip-in channel = middle_w.
        self.strided_up_im = DeconvResBlock(
            inner_w, middle_w, skip_in_ch=middle_w, stride=2
        )

        self.middle_dec = nn.ModuleList(
            [DeconvResBlock(middle_w, middle_w, skip_in_ch=middle_w, stride=1)
             for _ in range(m_middle)]
        )
        # Middle → outer (strided up). Pairs with strided_down_om;
        # skip-in channel = outer_w.
        self.strided_up_mo = DeconvResBlock(
            middle_w, outer_w, skip_in_ch=outer_w, stride=2
        )

        self.outer_dec = nn.ModuleList(
            [DeconvResBlock(outer_w, outer_w, skip_in_ch=outer_w, stride=1)
             for _ in range(n_outer)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: `[B, in_ch, H, W]`  →  returns `[B, outer_w, H, W]`.

        Skips are collected encoder-side in push order, then popped
        decoder-side in reverse. The pairing is: last encoder block's
        skip feeds the first decoder block (mirror-symmetric).
        """
        x = self.initial_conv(x)
        x = self.initial_gn(x)
        x = F.relu(x)

        skips: list[torch.Tensor] = []

        # Encoder: push one skip per ResBlock (including the strided ones).
        for block in self.outer_enc:
            x, skip = block(x)
            skips.append(skip)
        x, skip = self.strided_down_om(x)
        skips.append(skip)
        for block in self.middle_enc:
            x, skip = block(x)
            skips.append(skip)
        x, skip = self.strided_down_mi(x)
        skips.append(skip)
        for block in self.inner_enc:
            x, skip = block(x)
            skips.append(skip)

        # Decoder: pop in reverse — last-pushed pairs with first decoder block.
        for block in self.inner_dec:
            x = block(x, skips.pop())
        x = self.strided_up_im(x, skips.pop())
        for block in self.middle_dec:
            x = block(x, skips.pop())
        x = self.strided_up_mo(x, skips.pop())
        for block in self.outer_dec:
            x = block(x, skips.pop())

        # Defensive: every encoder skip should pair with exactly one
        # decoder consumer. If this fires, the encoder/decoder block
        # counts have drifted out of sync.
        assert not skips, f"unconsumed encoder skips: {len(skips)}"
        return x


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


class PolicyHead(nn.Module):
    """
    Per-cell directional policy head.

    Structure: small Pyramid Module (N=1, M=0 per design doc §4) → 3×3 conv
    projecting to 8 output channels = 4 directions × 2 splits.

    Output stays in NCHW `[B, 8, H, W]`. The caller (loss code at training
    time, inference code at eval time) does the layout transform
    `permute(0,2,3,1).reshape(B, -1)` to land at the cell-major flat layout
    that the action target uses (see `bc/actions.py` flat-layout pin and
    the cross-cutting permute contract in 5.18-3 session note).

    NESW direction order matches `bc.actions.{N,E,S,W}` and the
    sub-channel pin `sub = dir*2 + split`. Output channel 0 = N split=0,
    channel 1 = N split=1, channel 2 = E split=0, ... channel 7 = W split=1.
    """

    def __init__(self, in_ch: int):
        super().__init__()
        # Head Pyramid Module at uniform width. With M=0 at middle and
        # inner levels, the head still has its strided down/up structure
        # (no ResBlocks in between, just the strided pair at each level).
        self.head_pm = PyramidModule(
            in_ch=in_ch,
            n_outer=1, m_middle=0, m_inner=0,
            widths=(in_ch, in_ch, in_ch),
        )
        self.final_conv = nn.Conv2d(in_ch, 8, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → [B, 8, H, W]."""
        x = self.head_pm(x)
        x = self.final_conv(x)
        return x


class PassHead(nn.Module):
    """
    Scalar pass logit pooled from the trunk embedding.

    Structure: masked global avg pool over the unpadded region of the
    [B, C, H, W] trunk output → Linear(C, 1) → [B] scalar logit per sample.
    Loss code applies `binary_cross_entropy_with_logits` against `is_pass`.

    The mask is required because trunk activations at padded positions are
    non-zero (convs see the zero-padded input and produce output for the
    padded region). A plain AdaptiveAvgPool2d would average those into the
    pool and the per-sample dilution would scale with how much of the 32×32
    grid is real board — a board-size leak into the pass signal.
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.linear = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W], valid_mask: [B, 1, H, W] bool → [B] pre-sigmoid logit."""
        m = valid_mask.to(x.dtype)
        summed = (x * m).sum(dim=(2, 3))                # [B, C]
        count = m.sum(dim=(2, 3)).clamp(min=1.0)        # [B, 1]
        pooled = summed / count                         # [B, C]
        return self.linear(pooled).squeeze(-1)          # [B]


class ValueHead(nn.Module):
    """
    Categorical placement value head (8-way: 1st through 8th).

    Two architectural variants, selected by `variant`:

      "direct" (v1 spike choice — formerly "option B" in 5.20-2 / 5.20-3):

          x ─→ Conv2d(C → 1, k=3) → ReLU → mask → flatten → Linear(H·W, 8)

        Thin head — the Linear does all spatial integration. 3×3 receptive
        field per cell before flatten; ~9k params. Anomalously small relative
        to the policy head's ~1.1M.

      "pyramid" (full DeepNash spec — formerly "option C" in 5.20-2 / 5.20-3):

          x ─→ PyramidModule(N=0/M=0/M=0, uniform width=C)
            ─→ Conv2d(C → 1, k=3) → ReLU → mask → flatten → Linear(H·W, 8)

        Adds a shape-preserving U-Net-style head before proj_conv: 32→16→8
        encoder + symmetric decoder with skip connections. Each cell in the
        pre-flatten map encodes global board context via the 8×8 bottleneck,
        which is the right shape for a placement-prediction (global) task.
        ~+0.82M params; brings the value head into the same size class as
        the policy head.

    Common to both: the mask multiply between ReLU and flatten zeroes
    padded-cell contributions, so the Linear layer doesn't see per-game-
    varying junk at padded positions.
    """

    def __init__(
        self,
        in_ch: int,
        H: int,
        W: int,
        n_classes: int = 8,
        variant: str = "direct",
    ):
        super().__init__()
        if variant not in VALUE_HEAD_VARIANTS:
            raise ValueError(
                f"variant must be one of {VALUE_HEAD_VARIANTS}; got {variant!r}"
            )
        self.variant = variant
        if variant == "pyramid":
            self.pre = PyramidModule(
                in_ch=in_ch, n_outer=0, m_middle=0, m_inner=0,
                widths=(in_ch, in_ch, in_ch),
            )
        else:
            self.pre = nn.Identity()
        self.proj_conv = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)
        self.linear = nn.Linear(H * W, n_classes)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W], valid_mask: [B, 1, H, W] bool → [B, n_classes]."""
        x = self.pre(x)                     # [B, C, H, W] (PM or Identity)
        x = self.proj_conv(x)               # [B, 1, H, W]
        x = F.relu(x)
        x = x * valid_mask.to(x.dtype)      # zero padded contributions
        x = x.flatten(1)                    # [B, H·W]
        x = self.linear(x)                  # [B, n_classes]
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class BCModel(nn.Module):
    """
    Top-level behavioral-cloning model.

    Composes trunk + three heads. `forward` returns a dict so the loss
    code can address each head's output independently and the overfit
    harness can log per-component losses.
    """

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        # Plain attribute (not a submodule/buffer/param), so it adds no
        # state_dict keys — the checkpoint's `arch` key is written from it.
        self.cfg = cfg
        self.trunk = PyramidModule(
            in_ch=cfg.in_ch,
            n_outer=cfg.n_outer,
            m_middle=cfg.m_middle,
            m_inner=cfg.m_inner,
            widths=(cfg.outer_width, cfg.middle_width, cfg.inner_width),
        )
        self.policy_head = PolicyHead(in_ch=cfg.outer_width)
        self.pass_head = PassHead(in_ch=cfg.outer_width)
        self.value_head = ValueHead(
            in_ch=cfg.outer_width, H=cfg.H, W=cfg.W, variant=cfg.value_head_variant,
        )

    def forward(
        self, obs: torch.Tensor, valid_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        obs:        `[B, in_ch, H, W]`
        valid_mask: `[B, 1, H, W]` bool — True over the unpadded board region.

        Returns:
          - `policy_logits`: `[B, 8, H, W]`  (NCHW; loss code does the
            permute to cell-major flat layout and applies the per-cell
            legality mask)
          - `pass_logit`: `[B]`              (pre-sigmoid; masked pool)
          - `value_logits`: `[B, 8]`          (padded cells masked before flatten)
        """
        trunk_out = self.trunk(obs)
        return {
            "policy_logits": self.policy_head(trunk_out),
            "pass_logit": self.pass_head(trunk_out, valid_mask),
            "value_logits": self.value_head(trunk_out, valid_mask),
        }
