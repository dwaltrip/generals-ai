"""`BCModel.forward`'s output as a typed, validated value object.

The three core head outputs are typed fields. The variant-gated aux-head logits
are stored in the open `aux` dict, keyed by each spec's `output_key`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from training.bc.aux_heads import AuxHeadSpec


@dataclass(frozen=True)
class ModelOut:
    policy_logits: torch.Tensor  # [B, 8, H, W] NCHW
    pass_logit: torch.Tensor     # [B] pre-sigmoid
    value_logits: torch.Tensor   # [B, 8]
    # Variant-gated aux logits, keyed by each spec's `output_key` (e.g.
    # "elim_logits" / "next_elim_logits"). Empty for non-aux runs.
    aux: dict[str, torch.Tensor] = field(default_factory=dict)
    # The built aux-head specs this output carries — the single source of which
    # heads are live, read by the loss and the dump. Same name as its source,
    # `BCModel.active_aux_specs`, so the list is one concept end to end.
    active_aux_specs: tuple[AuxHeadSpec, ...] = ()

    def __post_init__(self) -> None:
        # Validate that `aux` holds all the logits expected from the active aux heads.
        for spec in self.active_aux_specs:
            assert spec.output_key in self.aux, (
                f"aux head {spec.name!r} declared but {spec.output_key!r} not emitted"
            )
