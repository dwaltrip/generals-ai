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

from training.bc.loss import LossConfig
from training.bc.model_config import build_model_cfg
from training.bc.train_cli import build_arg_parser, config_from_args
from training.bc.train_config import TrainConfig


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


def test_lambda_elim_requires_enabled_head(tmp_path: Path) -> None:
    """A loss weight on the elim head without the head built is a config error,
    not a silent no-op; the matching arch flag makes it legal. The cross-cutting
    rule lives on `TrainConfig` (it needs both arch and loss visible), so it fires
    even though `LossConfig` itself accepts the weight."""
    paths = {"manifest": Path("m"), "intermediate": Path("i"), "run_dir": tmp_path}
    with pytest.raises(ValueError, match="lambda_elim > 0 requires"):
        TrainConfig(**paths, loss=LossConfig(lambda_elim=0.1))
    # With the head enabled it constructs, and the weights coerce to a tuple.
    cfg = TrainConfig(
        **paths,
        arch=build_model_cfg(elim_head_variant="time_bin"),
        loss=LossConfig(lambda_elim=0.1, elim_bin_weights=[1.0, 2.0, 0.5]),
    )
    assert cfg.loss.lambda_elim == 0.1
    assert cfg.loss.elim_bin_weights == (1.0, 2.0, 0.5)


def test_loss_knobs_flat_and_nested_equivalent(tmp_path: Path) -> None:
    """Back-compat: loss knobs parse both flat at the top level (the pre-nesting
    shape) and nested under `loss:`, yielding the same resolved `LossConfig`."""
    run_dir = tmp_path / "runs" / "r"
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps(
        {"manifest": "m", "intermediate": "i",
         "lambda_value": 0.7, "value_target_tau": 0.6}
    ))
    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps(
        {"manifest": "m", "intermediate": "i",
         "loss": {"lambda_value": 0.7, "value_target_tau": 0.6}}
    ))
    flat_cfg = TrainConfig.from_file(flat, run_dir=run_dir)
    nested_cfg = TrainConfig.from_file(nested, run_dir=run_dir)
    assert flat_cfg.loss == nested_cfg.loss
    assert flat_cfg.loss.lambda_value == 0.7
    assert flat_cfg.loss.value_target_tau == 0.6
    assert flat_cfg.loss.mu_pass == 1.0       # unset knob → LossConfig default


def test_loss_knob_set_in_both_shapes_raises(tmp_path: Path) -> None:
    """A knob set both flat and under `loss:` is ambiguous — rejected at parse."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps(
        {"manifest": "m", "intermediate": "i",
         "lambda_value": 0.7, "loss": {"lambda_value": 0.8}}
    ))
    with pytest.raises(ValueError, match="both at the top level and under"):
        TrainConfig.from_file(p, run_dir=tmp_path / "r")


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
