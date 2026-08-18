from __future__ import annotations

import pytest

from training.bc.config import MetricsConfig
from training.bc.config.targets_config import targets_cfg_from
from training.bc.emit_spec import emit_spec_for_model, emit_spec_from
from training.bc.model_config import build_model_cfg


@pytest.mark.parametrize(
    ("variant", "requested", "expect_alive"),
    [
        (None, False, False),
        (None, True, True),
        ("time_bin", False, True),  # loss-time need wins regardless of request
        ("next_death", False, False),
        ("next_death", True, True),
    ],
)
def test_emit_spec_from(variant, requested, expect_alive):
    arch = build_model_cfg(elim_head_variant=variant)
    spec = emit_spec_from(arch, MetricsConfig(include_alive_mask=requested), emit_frame_info=True)
    assert spec.emit_alive_mask is expect_alive
    assert spec.obs is arch.obs
    assert spec.targets == targets_cfg_from(arch)
    assert spec.emit_frame_info is True
    assert spec.attach_sim_frame is False


@pytest.mark.parametrize(
    ("variant", "expect_alive"), [(None, False), ("time_bin", True), ("next_death", True)]
)
def test_emit_spec_for_model(variant, expect_alive):
    spec = emit_spec_for_model(build_model_cfg(elim_head_variant=variant), emit_frame_info=False)
    assert spec.emit_alive_mask is expect_alive
