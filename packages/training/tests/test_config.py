"""Tests for bc/config: resolve_config, migrate, and the Path serde helpers."""

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
from training.bc.model_config import build_model_cfg
from training.bc.obs_config import OBS_CONFIG_DEFAULTS
from training.bc.train_config import TrainConfig


def _config(tmp_path) -> TrainConfig:
    """A non-default TrainConfig touching arch, nested obs, loss, and all paths."""
    return TrainConfig(
        manifest=tmp_path / "m.json",
        intermediate=tmp_path / "i",
        run_dir=tmp_path / "run",
        epochs=3,
        lr=1e-3,
        seed=42,
        arch=build_model_cfg(
            value_head_variant="pyramid",
            value_head_skip_dropout2d=0.1,
            obs=replace(OBS_CONFIG_DEFAULTS, dense_history_n=10),
        ),
    )


def _stored_block(config: TrainConfig) -> dict:
    """Mimic the on-disk config block: asdict + version, paths stringified, then a
    JSON round-trip. The round-trip turns tuples into lists, exercising the
    list->tuple coercion that needs to happen when rebuilding the config.
    """
    raw = {**asdict(config), "config_version": CONFIG_VERSION}
    stringify_paths(raw)
    return json.loads(json.dumps(raw))


def test_resolve_config_round_trip(tmp_path):
    config = _config(tmp_path)
    assert resolve_config(_stored_block(config)) == config


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
