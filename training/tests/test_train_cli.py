"""Overlay behavior of the BC training CLI bridge.

`config_from_args` overlays only explicitly-passed flags onto `TrainConfig`'s
field defaults — the parser owns no defaults of its own. These tests guard
that contract: the no-knob path must equal the dataclass defaults, passed
flags must override (with the `value_head`->`value_head_variant` rename and
the `pin_memory` adapter), and unspecified knobs must fall through.
"""

from __future__ import annotations

from pathlib import Path

from bc.train_cli import build_arg_parser, config_from_args
from bc.train_config import TrainConfig


_REQUIRED = ["--manifest", "m.json", "--intermediate", "inter", "--out-dir", "runs"]


def _config(argv: list[str]) -> TrainConfig:
    return config_from_args(build_arg_parser().parse_args(argv))


def test_no_knobs_equals_trainconfig_defaults() -> None:
    """With no training flags passed, every knob falls through to the
    `TrainConfig` field default — the single source. Guards against
    re-introducing a duplicate parser default that silently diverges."""
    cfg = _config(_REQUIRED)
    ref = TrainConfig(manifest=Path("m.json"), intermediate=Path("inter"), run_dir=cfg.run_dir)
    assert cfg == ref
    # run_dir is composed as out_dir / <run-id>.
    assert cfg.run_dir.parent == Path("runs")


def test_passed_flags_override_and_map() -> None:
    cfg = _config(_REQUIRED + [
        "--epochs", "8",
        "--batch-size", "32",
        "--lr", "0.001",
        "--value-head", "pyramid",
        "--pin-memory", "true",
        "--skip-val",
    ])
    assert cfg.epochs == 8
    assert cfg.batch_size == 32
    assert cfg.lr == 0.001
    assert cfg.value_head_variant == "pyramid"  # flag name -> field name
    assert cfg.pin_memory is True               # 'true' -> True (adapter)
    assert cfg.skip_val is True
    # Unspecified knobs still default.
    assert cfg.weight_decay == 1e-4
    assert cfg.num_workers == 2


def test_pin_memory_adapter_values() -> None:
    assert _config(_REQUIRED + ["--pin-memory", "auto"]).pin_memory is None
    assert _config(_REQUIRED + ["--pin-memory", "true"]).pin_memory is True
    assert _config(_REQUIRED + ["--pin-memory", "false"]).pin_memory is False
