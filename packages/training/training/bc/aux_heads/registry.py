"""The aux-head registry: variant-string → spec, the single dispatch point.

`spec_for` resolves the variant string to a spec at construction.
`BCModel` builds the head(s) and keeps the resolved spec as `active_aux_specs`. 
The dataset workers also use `spec_for`, and the model isn't available.
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
