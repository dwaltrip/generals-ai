"""Aux-head specs: one object per variant-gated auxiliary head, behind a registry.

See `docs/2026-06/6.21-1-aux-head-spec-registry-refactor.md`.
"""

from __future__ import annotations

from training.bc.aux_heads.base import AuxHeadSpec, AuxLossResult
from training.bc.aux_heads.registry import REGISTRY, spec_for


__all__ = ["AuxHeadSpec", "AuxLossResult", "REGISTRY", "spec_for"]
