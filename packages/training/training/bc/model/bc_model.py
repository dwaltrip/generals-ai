"""
Behavioral-cloning model: trunk (Pyramid Module) + policy / pass / value heads.

The trunk is a 2-contraction U-Net "Pyramid Module" — the inherited DeepNash
architecture (Perolat 2022, arXiv:2206.15378, supplementary Fig. 7) at the
half-width 128/128/160 variant (see `bc/model/trunk.py`). Three heads read
off the trunk's [B, C, H, W] spatial embedding (see `bc/model/heads/`).

The model returns a `ModelOut` (see `model/output.py`) so the loss code can
address each head's output independently and the overfit harness can log
component-wise losses.

Two deviations from the inherited design (`network-architecture-design.md`)
are flagged in code:
  1. GroupNorm in place of LayerNorm — see `trunk._gn` for the reasoning.
  2. Inline value head (option B from 5.20-2 session) — flagged in
     `heads/value.py`. Extension path back to the full-spec DeepNash head
     is the "pyramid" variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from training.bc.aux_heads.base import AuxHeadSpec
from training.bc.aux_heads.registry import spec_for
from training.bc.model.heads.pass_head import PassHead
from training.bc.model.heads.policy import PolicyHead
from training.bc.model.heads.value import ValueHead
from training.bc.model.output import ModelOut
from training.bc.model.trunk import PyramidModule
from training.bc.model_config import MODEL_CONFIG_DEFAULTS, ModelConfig


class BCModel(nn.Module):
    """
    Top-level behavioral-cloning model.

    Composes trunk + three heads. `forward` returns a dict so the loss
    code can address each head's output independently and the overfit
    harness can log per-component losses.
    """

    def __init__(self, cfg: ModelConfig = MODEL_CONFIG_DEFAULTS):
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
            in_ch=cfg.outer_width,
            H=cfg.H, W=cfg.W,
            variant=cfg.value_head_variant,
            dropout2d_p=cfg.value_head_dropout2d,
            dropout_p=cfg.value_head_dropout,
            dropout2d_site=cfg.value_head_dropout2d_site,
            skip_dropout2d_p=cfg.value_head_skip_dropout2d,
        )
        # Arch-gated aux head: constructed only for an enabled variant, so an
        # off model (variant None) carries zero extra state_dict keys and every
        # pre-elim checkpoint still loads strict=True. Assigning `None` registers
        # no submodule. The aux-head spec builds the head and names its output key
        # (`elim_logits` for time_bin, `next_elim_logits` for next_death).
        spec = spec_for(cfg.elim_head_variant)
        # The active aux-head specs this model built — the single owner of "which
        # aux heads are live." The loss, accumulator, and eval read this instead of
        # re-resolving the variant or sniffing output keys. Empty when no head is
        # built; one element today (the variants are mutually exclusive) but a tuple
        # so the additive-ready loss/accumulator loops iterate a uniform thing.
        self.active_aux_specs: tuple[AuxHeadSpec, ...] = (
            (spec,) if spec is not None else ()
        )
        self.elim_output_key: str | None = spec.output_key if spec is not None else None
        self.elim_head: nn.Module | None = (
            spec.build_head(cfg) if spec is not None else None
        )

    def forward(
        self, obs: torch.Tensor, valid_mask: torch.Tensor
    ) -> ModelOut:
        """
        obs:        `[B, in_ch, H, W]`
        valid_mask: `[B, 1, H, W]` bool — True over the unpadded board region.

        Returns a `ModelOut` with:
          - `policy_logits`: `[B, 8, H, W]`  (NCHW; loss code does the
            permute to cell-major flat layout and applies the per-cell
            legality mask)
          - `pass_logit`: `[B]`              (pre-sigmoid; masked pool)
          - `value_logits`: `[B, 8]`          (padded cells masked before flatten)
          - `aux`: the active aux head's logits keyed by `output_key` —
            `elim_logits` `[B, 8, n_bins]` for `time_bin`, `next_elim_logits`
            `[B, 8]` for `next_death`. Empty for a non-aux run (the variants are
            mutually exclusive; at most one head is built).
        """
        trunk_out = self.trunk(obs)
        aux: dict[str, torch.Tensor] = {}
        if self.elim_head is not None:
            assert self.elim_output_key is not None  # set together with elim_head
            aux[self.elim_output_key] = self.elim_head(trunk_out, valid_mask)
        return ModelOut(
            policy_logits=self.policy_head(trunk_out),
            pass_logit=self.pass_head(trunk_out, valid_mask),
            value_logits=self.value_head(trunk_out, valid_mask),
            aux=aux,
            active_aux_specs=self.active_aux_specs,
        )
