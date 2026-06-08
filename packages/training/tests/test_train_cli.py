"""The `--config` contract of the BC training CLI bridge.

`config_from_args` builds a fresh-run `TrainConfig` from a `--config` JSON file
(arch + recipe + data paths, possibly partial over defaults) plus the
operational CLI flags. These tests guard that contract: file values land,
unset fields fall through to defaults, operational flags overlay, the
`pin_memory` adapter maps, and the required-flag / resume-only-directive
guards raise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bc.train_cli import build_arg_parser, config_from_args
from bc.train_config import TrainConfig


def _write_config(tmp_path: Path, **fields: object) -> Path:
    """A --config file with the required data paths plus any extra fields."""
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"manifest": "m.json", "intermediate": "inter", **fields}))
    return p


def _config(argv: list[str]) -> TrainConfig:
    return config_from_args(build_arg_parser().parse_args(argv))


def test_config_file_over_defaults(tmp_path: Path) -> None:
    """File values land; unset recipe/arch fields fall through to defaults."""
    cfg_path = _write_config(
        tmp_path, epochs=8, batch_size=32, lr=0.001, gpu="A100",
        arch={"value_head_variant": "pyramid"},
    )
    cfg = _config(["--config", str(cfg_path), "--out-dir", str(tmp_path / "runs")])

    assert cfg.epochs == 8
    assert cfg.batch_size == 32
    assert cfg.lr == 0.001
    assert cfg.gpu == "A100"
    assert cfg.arch.value_head_variant == "pyramid"
    assert cfg.arch.outer_width == 128       # partial arch → other fields default
    assert cfg.weight_decay == 1e-4          # unset recipe → default
    assert cfg.manifest == Path("m.json")
    assert cfg.run_dir.parent == tmp_path / "runs"


def test_operational_flags_overlay(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    cfg = _config([
        "--config", str(cfg_path), "--out-dir", str(tmp_path / "runs"),
        "--num-workers", "0", "--pin-memory", "true", "--skip-val",
        "--max-batches", "5",
    ])
    assert cfg.num_workers == 0
    assert cfg.pin_memory is True            # 'true' -> True (adapter)
    assert cfg.skip_val is True
    assert cfg.max_batches == 5
    assert cfg.prefetch_factor == 2          # unset operational → default


def test_pin_memory_adapter_values(tmp_path: Path) -> None:
    base = ["--config", str(_write_config(tmp_path)), "--out-dir", str(tmp_path / "runs")]
    assert _config(base + ["--pin-memory", "auto"]).pin_memory is None
    assert _config(base + ["--pin-memory", "true"]).pin_memory is True
    assert _config(base + ["--pin-memory", "false"]).pin_memory is False


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _config(["--out-dir", str(tmp_path / "runs")])


def test_resume_only_directive_without_resume_raises(tmp_path: Path) -> None:
    base = ["--config", str(_write_config(tmp_path)), "--out-dir", str(tmp_path / "runs")]
    with pytest.raises(SystemExit):
        _config(base + ["--force-config-mismatch"])
    with pytest.raises(SystemExit):
        _config(base + ["--legacy-lr-warmup-batches", "500"])
