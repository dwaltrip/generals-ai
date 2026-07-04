"""Tests for `bc.config`."""

from __future__ import annotations

from dataclasses import asdict, replace
import json

import pytest

from training.bc.config import (
    CONFIG_VERSION,
    migrate,
    resolve_config,
    rewrap_paths,
    stringify_paths,
)
from training.bc.model_config import MODEL_CONFIG_DEFAULTS, ModelConfig, build_model_cfg
from training.bc.obs_config import OBS_CONFIG_DEFAULTS
from training.bc.train_config import TrainConfig


def _config(tmp_path, arch: ModelConfig = MODEL_CONFIG_DEFAULTS) -> TrainConfig:
    return TrainConfig(
        manifest=tmp_path / "m.json",
        intermediate=tmp_path / "i",
        run_dir=tmp_path / "run",
        epochs=3,
        lr=1e-3,
        seed=42,
        arch=arch,
    )


def test_resolve_config_json_round_trip(tmp_path):
    # Non-default arch, so equality can't pass by refilling from defaults.
    arch = build_model_cfg(
        value_head_variant="pyramid",
        value_head_skip_dropout2d=0.1,
        obs=replace(OBS_CONFIG_DEFAULTS, dense_history_n=10),
    )
    config = _config(tmp_path, arch=arch)
    raw = {**asdict(config), "config_version": CONFIG_VERSION}
    stringify_paths(raw)
    json_round_tripped = json.loads(json.dumps(raw))
    assert resolve_config(json_round_tripped) == config


def test_migrate_rejects_future_version():
    with pytest.raises(ValueError, match="exceeds supported"):
        migrate({"config_version": CONFIG_VERSION + 1})


def test_migrate_is_noop_copy_at_current_version():
    block = {"config_version": CONFIG_VERSION, "lr": 1e-3}
    out = migrate(block)
    assert out == block
    assert out is not block


def test_stringify_rewrap_symmetry(tmp_path):
    original = {**asdict(_config(tmp_path)), "config_version": CONFIG_VERSION}
    data = dict(original)
    stringify_paths(data)
    assert all(isinstance(data[k], str) for k in ("manifest", "intermediate", "run_dir"))
    rewrap_paths(data)
    assert data == original
