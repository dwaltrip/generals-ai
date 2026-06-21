"""The aux-head registry: variant-string → spec, the single dispatch point.

Pipeline sites select a spec two ways. Construction-time sites (model, dataset,
eval) key on `arch.elim_head_variant` via `spec_for`. The loss and dump, which
receive only `model_out`/`out` and no arch, instead iterate `REGISTRY` and act on
the spec whose `output_key` is present — preserving the existing "no-op unless the
head emitted its logits" gating.
"""

from __future__ import annotations

from training.bc.aux_heads.base import AuxHeadSpec, AuxLossResult
from training.bc.aux_heads.next_death import NextDeathSpec
from training.bc.aux_heads.time_bin import TimeBinSpec


TIME_BIN_SPEC = TimeBinSpec()
NEXT_DEATH_SPEC = NextDeathSpec()

# Insertion order is the public order (e.g. the `model_config` validation error
# message). Keep `time_bin` first to match the historical `ELIM_HEAD_VARIANTS`.
REGISTRY: dict[str, AuxHeadSpec] = {
    TIME_BIN_SPEC.name: TIME_BIN_SPEC,
    NEXT_DEATH_SPEC.name: NEXT_DEATH_SPEC,
}


def spec_for(variant: str | None) -> AuxHeadSpec | None:
    """The spec for a variant string, or `None` when no head is enabled.

    Raises `KeyError` on an unknown variant — `model_config` validates the string
    against `REGISTRY` upstream, so a bad value here is a programming error.
    """
    if variant is None:
        return None
    return REGISTRY[variant]


__all__ = ["AuxHeadSpec", "AuxLossResult", "REGISTRY", "spec_for"]
