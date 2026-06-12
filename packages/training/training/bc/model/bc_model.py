"""
Behavioral-cloning model: trunk (Pyramid Module) + policy / pass / value heads.

The trunk is a 2-contraction U-Net "Pyramid Module" — the inherited DeepNash
architecture (Perolat 2022, arXiv:2206.15378, supplementary Fig. 7) at the
half-width 128/128/160 variant (see `bc/model/trunk.py`). Three heads read
off the trunk's [B, C, H, W] spatial embedding (see `bc/model/heads/`).

The model returns a dict so the loss code can address each head's output
independently and the overfit harness can log component-wise losses.

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

from training.bc.model.heads.pass_head import PassHead
from training.bc.model.heads.policy import PolicyHead
from training.bc.model.heads.value import ValueHead
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
            in_ch=cfg.outer_width, H=cfg.H, W=cfg.W, variant=cfg.value_head_variant,
            dropout2d_p=cfg.value_head_dropout2d, dropout_p=cfg.value_head_dropout,
            dropout2d_site=cfg.value_head_dropout2d_site,
            skip_dropout2d_p=cfg.value_head_skip_dropout2d,
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
