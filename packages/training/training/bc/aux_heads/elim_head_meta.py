"""Torch-free metadata for the elim aux-head family."""

from enum import StrEnum


class ElimHeadVariant(StrEnum):
    TIME_BIN = "time_bin"
    NEXT_DEATH = "next_death"

    @classmethod
    def coerce(cls, raw: str | ElimHeadVariant | None) -> ElimHeadVariant | None:
        return None if raw is None else cls(raw)


def does_elim_head_need_alive_mask(variant: str | ElimHeadVariant | None) -> bool:
    return variant == ElimHeadVariant.TIME_BIN
