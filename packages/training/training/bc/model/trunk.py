"""
Pyramid Module trunk — the DeepNash 2-contraction U-Net (Perolat 2022,
arXiv:2206.15378, supplementary Fig. 7) and its residual building blocks.

    32² → 16² → 8² → 16² → 32²    (contract-bottleneck-expand)

`PyramidModule` is used both as the main trunk and as the small embedded
"head Pyramid Module" inside the policy / pyramid-variant value heads
(see `bc/model/heads/`).

One deviation from the inherited design (`network-architecture-design.md`)
lives here: GroupNorm in place of LayerNorm — see `_gn` for the reasoning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        skip_dropout2d_p: float = 0.0,
    ):
        super().__init__()
        outer_w, middle_w, inner_w = widths

        # Channel dropout on skip connections at consumption time (train-time
        # only; p=0 ≡ identity, so the main trunk / policy-head instances are
        # unaffected unless they opt in). Skips carry full-resolution features
        # around the bottleneck — dropping them stochastically forces the
        # decoder to lean on the compressed global route instead. One shared
        # module (no params); each consumption site draws its own mask.
        self.skip_dropout = nn.Dropout2d(skip_dropout2d_p)

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
        # Skips pass through `skip_dropout` at consumption (identity at p=0).
        for block in self.inner_dec:
            x = block(x, self.skip_dropout(skips.pop()))
        x = self.strided_up_im(x, self.skip_dropout(skips.pop()))
        for block in self.middle_dec:
            x = block(x, self.skip_dropout(skips.pop()))
        x = self.strided_up_mo(x, self.skip_dropout(skips.pop()))
        for block in self.outer_dec:
            x = block(x, self.skip_dropout(skips.pop()))

        # Defensive: every encoder skip should pair with exactly one
        # decoder consumer. If this fires, the encoder/decoder block
        # counts have drifted out of sync.
        assert not skips, f"unconsumed encoder skips: {len(skips)}"
        return x
