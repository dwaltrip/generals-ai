"""Smoke tests for the aux-head registry — dispatch wiring only.

The specs' loss/encode/dump bodies are the same code exercised by `test_loss.py`
and `test_dataset.py`; this file pins the registry lookup and the spec/output-key
mapping, not the numerics.
"""

from __future__ import annotations

import pytest

from training.bc.aux_heads import REGISTRY, spec_for
from training.bc.aux_heads.base import AuxHeadSpec


def test_registry_holds_the_two_variants():
    assert tuple(REGISTRY) == ("time_bin", "next_death")
    for name, spec in REGISTRY.items():
        assert spec.name == name
        assert isinstance(spec, AuxHeadSpec)


def test_registry_matches_model_config_variant_list():
    # `model_config` is deliberately torch-free, so it can't import the registry;
    # `ELIM_HEAD_VARIANTS` stays a literal there. This pins the two in sync so a
    # registry-only addition can't silently diverge from the config validation.
    from training.bc.model_config import ELIM_HEAD_VARIANTS

    assert tuple(REGISTRY) == ELIM_HEAD_VARIANTS


def test_output_keys_are_distinct_and_correct():
    assert REGISTRY["time_bin"].output_key == "elim_logits"
    assert REGISTRY["next_death"].output_key == "next_elim_logits"


def test_spec_for_dispatch():
    assert spec_for("time_bin") is REGISTRY["time_bin"]
    assert spec_for("next_death") is REGISTRY["next_death"]
    assert spec_for(None) is None


def test_spec_for_unknown_variant_raises():
    # model_config validates the string upstream; an unknown value is a bug.
    with pytest.raises(KeyError):
        spec_for("not_a_variant")
