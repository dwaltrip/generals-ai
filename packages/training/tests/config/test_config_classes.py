from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from training.bc.config import TargetsConfig
from training.bc.config.metrics_config import metrics_cfg_from
from training.bc.config.targets_config import targets_cfg_from
from training.bc.model_config import build_model_cfg
from training.bc.train_config import TrainConfig


EDGES = (10, 20, 40)


def test_targets_config_validation():
    with pytest.raises(ValueError, match="time_bin requires"):
        TargetsConfig(elim_variant="time_bin", elim_bin_edges=None)
    with pytest.raises(ValueError, match="requires the time_bin variant"):
        TargetsConfig(elim_variant="next_death", elim_bin_edges=EDGES)
    with pytest.raises(ValueError, match="requires the time_bin variant"):
        TargetsConfig(elim_variant=None, elim_bin_edges=EDGES)

    # bin edges are lists in the raw dict and should be coerced to tuples on init.
    cfg = TargetsConfig(
        elim_variant="time_bin",
        elim_bin_edges=list(EDGES),  # pyright: ignore[reportArgumentType]
    )
    assert cfg.elim_bin_edges == EDGES
    assert isinstance(cfg.elim_bin_edges, tuple)


@pytest.mark.parametrize(
    ("variant", "expect_edges"),
    [(None, None), ("time_bin", "arch"), ("next_death", None)],
)
def test_targets_cfg_from(variant, expect_edges):
    arch = build_model_cfg(elim_head_variant=variant)
    cfg = targets_cfg_from(arch)
    assert cfg.elim_variant == variant
    assert cfg.elim_bin_edges == (arch.elim_bin_edges if expect_edges == "arch" else None)


@pytest.mark.parametrize(
    ("variant", "expect_request"),
    [(None, False), ("time_bin", True), ("next_death", True)],
)
def test_metrics_cfg_from(tmp_path, variant, expect_request):
    train_cfg = TrainConfig(
        manifest=tmp_path / "m.json",
        intermediate=tmp_path / "i",
        run_dir=tmp_path / "r",
        arch=build_model_cfg(elim_head_variant=variant),
    )
    assert metrics_cfg_from(train_cfg).include_alive_mask is expect_request


def test_config_package_imports_torch_free():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import training.bc.config, training.bc.emit_spec, sys; "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
        cwd=Path(__file__).parent,
    )
