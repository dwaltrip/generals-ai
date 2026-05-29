"""Unit tests for resume bookkeeping in `bc.run_dir` / `bc.resume`.

Covers the pure logic — suffix discovery, checkpoint-format detection, the
legacy cold-restart gate, the drift gate's buckets, and the parent-config
overlay. The end-to-end resume flow (run_training wiring, checkpoint restore,
suffixed artifacts) is exercised by a manual local run.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from bc.resume import _resume_config, bc_resume
from bc.run_dir import ResumeInfo, check_drift, prepare_resume
from bc.train_config import TrainConfig, json_default


def _config(**overrides: object) -> TrainConfig:
    base: dict = dict(manifest=Path("m.json"), intermediate=Path("inter"), run_dir=Path("runs/r"))
    base.update(overrides)
    return TrainConfig(**base)


def _save_ckpt(path: Path, *, legacy: bool) -> None:
    """Write a tiny loadable checkpoint. `prepare_resume` now peeks the latest
    checkpoint's structure, so fixtures need real `torch.save` payloads (not
    empty `.touch()` files). Combined carries a `"model"` key; legacy is a bare
    state_dict (param-name keys)."""
    obj = {"model": {}, "epoch": 0} if not legacy else {"trunk.0.weight": torch.zeros(1)}
    torch.save(obj, path)


def _make_run(
    tmp_path: Path,
    epochs: list[int],
    resume_artifacts: tuple[str, ...] = (),
    legacy: bool = False,
) -> Path:
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    for e in epochs:
        _save_ckpt(ckpts / f"epoch_{e:03d}.pt", legacy=legacy)
    (tmp_path / "args.json").touch()
    for name in resume_artifacts:
        (tmp_path / name).touch()
    return tmp_path


def test_prepare_resume_first(tmp_path: Path) -> None:
    run = _make_run(tmp_path, [1, 2])
    info = prepare_resume(run)
    assert info.next_suffix == "_resume_01"
    assert info.latest_checkpoint.name == "epoch_002.pt"
    assert info.parent_epoch == 2
    assert info.parent_args_path == run / "args.json"
    assert info.is_legacy_checkpoint is False


def test_prepare_resume_second(tmp_path: Path) -> None:
    run = _make_run(
        tmp_path,
        [1, 2, 3],
        resume_artifacts=("args_resume_01.json", "run_resume_01.log", "batches_resume_01.jsonl"),
    )
    info = prepare_resume(run)
    assert info.next_suffix == "_resume_02"
    assert info.parent_epoch == 3
    assert info.parent_args_path == run / "args_resume_01.json"


def test_prepare_resume_detects_legacy(tmp_path: Path) -> None:
    run = _make_run(tmp_path, [1, 2], legacy=True)
    assert prepare_resume(run).is_legacy_checkpoint is True


def test_prepare_resume_no_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir()
    with pytest.raises(SystemExit):
        prepare_resume(tmp_path)


def test_prepare_resume_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        prepare_resume(tmp_path / "nope")


def test_check_drift_excluded_is_free() -> None:
    parent = asdict(_config(epochs=2))
    # epochs (the resume target) differs — excluded, so no abort.
    check_drift(_config(epochs=8), parent, force_config_mismatch=False)


def test_check_drift_trajectory_gated_by_force() -> None:
    parent = asdict(_config(batch_size=64))
    with pytest.raises(SystemExit):
        check_drift(_config(batch_size=128), parent, force_config_mismatch=False)
    check_drift(_config(batch_size=128), parent, force_config_mismatch=True)  # bypassed


def test_check_drift_lr_locked_even_with_force() -> None:
    parent = asdict(_config(lr=3e-4))
    with pytest.raises(SystemExit):
        check_drift(_config(lr=1e-3), parent, force_config_mismatch=True)


def test_resume_config_overlay() -> None:
    parent = asdict(_config(batch_size=128, lr=1e-4, seed=7, epochs=5))
    base = _config(run_dir=Path("runs/resumed"))  # env paths + the resume dir
    effective = _resume_config(base, parent, overrides={"epochs": 10})
    assert effective.batch_size == 128             # carried from parent
    assert effective.lr == 1e-4                    # carried from parent
    assert effective.seed == 7                     # carried from parent
    assert effective.epochs == 10                  # explicit override wins
    assert effective.run_dir == Path("runs/resumed")  # path from base


# --- legacy cold-restart gate (bc_resume) ---
# These assert the gate fires *before* any training setup. The earlier checks
# (parent-args load, drift, epoch target) are arranged to pass so the abort is
# attributable to the legacy/flag mismatch, not an unrelated failure.

def _gate_run(tmp_path: Path) -> tuple[TrainConfig, Path]:
    """A run dir with a non-drifting parent args.json + a config to resume it
    (parent_epoch 2, epochs 5 → past the epoch-target check)."""
    run = tmp_path
    parent = _config(run_dir=run, epochs=5)
    (run / "args.json").write_text(json.dumps(asdict(parent), default=json_default))
    return _config(run_dir=run, epochs=5), run


def _info(run: Path, *, legacy: bool) -> ResumeInfo:
    return ResumeInfo(
        next_suffix="_resume_01",
        latest_checkpoint=run / "checkpoints" / "epoch_002.pt",
        parent_epoch=2,
        parent_args_path=run / "args.json",
        is_legacy_checkpoint=legacy,
    )


def test_bc_resume_aborts_on_legacy_without_flag(tmp_path: Path) -> None:
    config, run = _gate_run(tmp_path)
    with pytest.raises(SystemExit, match="legacy"):
        bc_resume(config, force_config_mismatch=False, overrides={},
                  info=_info(run, legacy=True), legacy_lr_warmup_batches=None)


def test_bc_resume_aborts_on_combined_with_flag(tmp_path: Path) -> None:
    config, run = _gate_run(tmp_path)
    with pytest.raises(SystemExit, match="combined-format"):
        bc_resume(config, force_config_mismatch=False, overrides={},
                  info=_info(run, legacy=False), legacy_lr_warmup_batches=500)
