"""Checkpoint read/write tests. A few notable cases:

    - Legacy (v0) arch reconstruction
    - The v1 versioned format
    - The on-disk format discriminators
"""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest
import torch

from training.bc.checkpoint import (
    LEGACY_ARCH,
    LEGACY_OBS_CFG,
    is_arch_bearing,
    is_legacy_checkpoint,
)
from training.bc.config import CONFIG_VERSION
from training.bc.inference import BCModelHandle
from training.bc.model import BCModel
from training.bc.model_config import MODEL_CONFIG_DEFAULTS, ModelConfig, build_model_cfg
from training.bc.obs_config import OBS_CONFIG_DEFAULTS
from training.bc.state import TrainingState
from training.bc.storage.checkpoint import load_checkpoint, serialize_checkpoint
from training.bc.train_config import TrainConfig


# TODO(ckpt-cfg-refactor-note): We need to rewrite these tests at some point.
# Revisit later in the ckpt-config refactor (or after).

def test_load_bc_model_bare_state_dict(tmp_path):
    """A legacy bare state_dict (no `arch` key) reconstructs via LEGACY_ARCH."""
    device = torch.device("cpu")
    # Build the source at the legacy architecture — the live default has since
    # diverged (fp16 obs, player-status channels), so the bare state_dict has to
    # be legacy-shaped to round-trip through the LEGACY_ARCH fallback.
    src = BCModel(LEGACY_ARCH)
    ckpt = tmp_path / "epoch_001.pt"
    torch.save(src.state_dict(), ckpt)

    loaded = load_checkpoint(ckpt, device).model

    assert isinstance(loaded, BCModel)
    assert loaded.cfg == LEGACY_ARCH       # legacy fallback, default variant
    assert not loaded.training             # returned in eval mode
    src_sd, loaded_sd = src.state_dict(), loaded.state_dict()
    assert src_sd.keys() == loaded_sd.keys()
    for k in src_sd:
        assert torch.equal(src_sd[k], loaded_sd[k].cpu()), f"param mismatch: {k}"


def test_load_legacy_honors_value_head_variant_arg(tmp_path):
    """A legacy pyramid-trained checkpoint loads only when the load-time variant
    matches — strict load is the backstop. This is the unit-level twin of the
    parity fingerprint (which loads a real pyramid legacy checkpoint)."""
    device = torch.device("cpu")
    src = BCModel(replace(LEGACY_ARCH, value_head_variant="pyramid"))
    ckpt = tmp_path / "epoch_001.pt"
    torch.save(src.state_dict(), ckpt)  # bare state_dict — legacy

    loaded = load_checkpoint(ckpt, device, value_head_variant="pyramid").model
    assert loaded.cfg == replace(LEGACY_ARCH, value_head_variant="pyramid")


def test_arch_bearing_checkpoint_reconstructs_nondefault(tmp_path):
    """The round-trip the refactor exists for: a non-default arch is recorded in
    the checkpoint's `arch` key and reconstructed authoritatively on load (the
    load-time variant arg is ignored), with strict weight load."""
    device = torch.device("cpu")
    cfg = build_model_cfg(outer_width=64, middle_width=64, inner_width=96)
    src = BCModel(cfg)
    ckpt = tmp_path / "epoch_002.pt"
    # Mimic TrainingState.save's combined format (model + arch).
    torch.save({"model": src.state_dict(), "arch": asdict(cfg), "epoch": 2}, ckpt)

    loaded = load_checkpoint(ckpt, device, value_head_variant="direct").model
    assert loaded.cfg == cfg               # arch key authoritative
    src_sd, loaded_sd = src.state_dict(), loaded.state_dict()
    assert src_sd.keys() == loaded_sd.keys()
    for k in src_sd:
        assert torch.equal(src_sd[k], loaded_sd[k].cpu()), f"param mismatch: {k}"


def test_arch_in_ch_checksum_mismatch_raises(tmp_path):
    """A recorded in_ch contradicting the obs channel count (an obs-channel-
    formula drift) raises a clear error on load, not a cryptic shape mismatch."""
    cfg = build_model_cfg()
    src = BCModel(cfg)
    ckpt = tmp_path / "epoch_001.pt"
    arch = {**asdict(cfg), "in_ch": cfg.in_ch + 1}  # tampered checksum
    torch.save({"model": src.state_dict(), "arch": arch, "epoch": 1}, ckpt)

    with pytest.raises(ValueError, match="contradicts obs"):
        load_checkpoint(ckpt, torch.device("cpu"))


def test_default_arch_is_no_op_against_bare_model(tmp_path):
    """`build_model_cfg()` yields the canonical default (`MODEL_CONFIG_DEFAULTS`),
    so a bare `BCModel()` and one built from the factory default match (the smoke
    fingerprint pins the values)."""
    assert build_model_cfg() == MODEL_CONFIG_DEFAULTS
    assert BCModel().state_dict().keys() == BCModel(build_model_cfg()).state_dict().keys()


def test_is_legacy_checkpoint_detection(tmp_path):
    legacy = tmp_path / "legacy.pt"
    combined = tmp_path / "combined.pt"
    torch.save(BCModel().state_dict(), legacy)            # bare state_dict
    torch.save({"model": {}, "optim": {}, "epoch": 3}, combined)  # combined wrapper

    assert is_legacy_checkpoint(legacy) is True
    assert is_legacy_checkpoint(combined) is False


# --- obs sub-config back-fill: load → legacy, fresh → live default ---
#
# When a new ObsConfig field is added (obs_dtype is the first), a checkpoint's
# `obs` block predates it. The load path must back-fill the missing key from the
# FROZEN legacy snapshot, while fresh construction fills from the LIVE defaults.
# These tests force `DEFAULTS != LEGACY` so they discriminate regardless of what
# the current live default happens to be.


def test_load_missing_obs_dtype_uses_legacy(tmp_path, monkeypatch):
    """A checkpoint whose `obs` block lacks `obs_dtype` back-fills it from the
    legacy snapshot, not the live default — so flipping the default never
    silently re-describes an old checkpoint's obs precision."""
    monkeypatch.setattr(
        "training.bc.model_config.OBS_CONFIG_DEFAULTS",
        replace(OBS_CONFIG_DEFAULTS, obs_dtype="fp16"),
    )
    cfg = build_model_cfg()
    src = BCModel(cfg)
    arch = asdict(cfg)
    del arch["obs"]["obs_dtype"]  # simulate a pre-obs_dtype checkpoint
    ckpt = tmp_path / "epoch_001.pt"
    torch.save({"model": src.state_dict(), "arch": arch, "epoch": 1}, ckpt)

    loaded = load_checkpoint(ckpt, torch.device("cpu")).model
    assert loaded.cfg.obs.obs_dtype == LEGACY_OBS_CFG.obs_dtype  # legacy, not the fp16 default


def test_load_preserves_explicit_obs_dtype(tmp_path):
    """An `obs_dtype` explicitly recorded in a checkpoint's arch survives load —
    neither the legacy back-fill nor the default merge overrides it."""
    cfg = build_model_cfg()
    src = BCModel(cfg)
    arch = asdict(cfg)
    arch["obs"]["obs_dtype"] = "fp16"
    ckpt = tmp_path / "epoch_001.pt"
    torch.save({"model": src.state_dict(), "arch": arch, "epoch": 1}, ckpt)

    loaded = load_checkpoint(ckpt, torch.device("cpu")).model
    assert loaded.cfg.obs.obs_dtype == "fp16"


def test_load_missing_toplevel_key_uses_legacy(tmp_path, monkeypatch):
    """The top-level twin of the obs_dtype test: an arch dict predating a
    top-level ModelConfig field (value_head_dropout is the first such
    addition) back-fills it from LEGACY_ARCH, not the live default."""
    monkeypatch.setattr(
        "training.bc.model_config.MODEL_CONFIG_DEFAULTS",
        replace(MODEL_CONFIG_DEFAULTS, value_head_dropout=0.5),
    )
    cfg = build_model_cfg(value_head_dropout=0.0)
    src = BCModel(cfg)
    arch = asdict(cfg)
    del arch["value_head_dropout"]    # simulate a pre-field checkpoint
    del arch["value_head_dropout2d"]
    ckpt = tmp_path / "epoch_001.pt"
    torch.save({"model": src.state_dict(), "arch": arch, "epoch": 1}, ckpt)

    loaded = load_checkpoint(ckpt, torch.device("cpu")).model
    assert loaded.cfg.value_head_dropout == LEGACY_ARCH.value_head_dropout  # 0.0, not 0.5


def test_load_pre_elim_checkpoint_backfills_disabled(tmp_path):
    """A checkpoint whose `arch` predates the elim-head fields back-fills them
    from LEGACY_ARCH (head off) and loads strict=True with an identical key set —
    the entire reason the head is arch-gated. Guards the backward-compat
    contract: existing checkpoints keep loading after the head lands."""
    cfg = build_model_cfg()
    src = BCModel(cfg)
    arch = asdict(cfg)
    del arch["elim_head_variant"]   # simulate a pre-elim checkpoint
    del arch["elim_bin_edges"]
    ckpt = tmp_path / "epoch_001.pt"
    torch.save({"model": src.state_dict(), "arch": arch, "epoch": 1}, ckpt)

    loaded = load_checkpoint(ckpt, torch.device("cpu")).model
    assert loaded.cfg.elim_head_variant is None
    assert loaded.cfg.elim_bin_edges == LEGACY_ARCH.elim_bin_edges
    # No elim params on either side, and strict load already passed in
    # load_checkpoint — assert key parity explicitly as the backstop.
    assert src.state_dict().keys() == loaded.state_dict().keys()
    assert not any("elim" in k for k in loaded.state_dict())


def test_load_checkpoint_migrates_retired_elim_head_enabled(tmp_path):
    """A checkpoint from the `elim_head_enabled` era (recorded the now-retired
    bool, no `elim_head_variant`) migrates: `enabled=True` → `"time_bin"`. Guards
    the `arch_for_load` translation shim that keeps those checkpoints loadable
    after the field was folded into `elim_head_variant`."""
    cfg = build_model_cfg(elim_head_variant="time_bin")
    src = BCModel(cfg)
    arch = asdict(cfg)
    arch.pop("elim_head_variant")
    arch["elim_head_enabled"] = True   # simulate the retired-field era
    ckpt = tmp_path / "epoch_001.pt"
    torch.save({"model": src.state_dict(), "arch": arch, "epoch": 1}, ckpt)

    loaded = load_checkpoint(ckpt, torch.device("cpu")).model
    assert loaded.cfg.elim_head_variant == "time_bin"
    assert src.state_dict().keys() == loaded.state_dict().keys()


def test_fresh_partial_obs_uses_live_default(monkeypatch):
    """A fresh config from a partial `obs` dict (not a checkpoint load) fills
    `obs_dtype` from the live defaults — the complement of the load-path
    back-fill above."""
    monkeypatch.setattr(
        "training.bc.model_config.OBS_CONFIG_DEFAULTS",
        replace(OBS_CONFIG_DEFAULTS, obs_dtype="fp16"),
    )
    cfg = build_model_cfg(obs={"dense_history_n": 5})
    assert cfg.obs.obs_dtype == "fp16"  # live default, not legacy fp32


# --- v1 versioned format: serialize_checkpoint / load_checkpoint ---


def _v1_config(tmp_path, arch: ModelConfig | None = None) -> TrainConfig:
    """A valid TrainConfig with paths under tmp_path."""
    return TrainConfig(
        manifest=tmp_path / "m.json",
        intermediate=tmp_path / "i",
        run_dir=tmp_path / "run",
        arch=arch if arch is not None else build_model_cfg(),
    )


def _runtime(state: TrainingState) -> dict:
    """The runtime-state portion of a checkpoint, as serialize_checkpoint expects."""
    return {
        "model": state.model.state_dict(),
        "optim": state.optim.state_dict(),
        "scaler": state.scaler.state_dict(),
        "epoch": state.epoch,
    }


def test_v1_round_trip(tmp_path):
    """Test v1 config round trip: serialize, save, load, and deserialize.
    Config values (Paths included), arch, and weights should match."""
    device = torch.device("cpu")
    # Use a non-default arch: if the loader ignored the stored arch and built
    # the model from current defaults, a default arch here would hide that bug.
    config = _v1_config(
        tmp_path,
        arch=build_model_cfg(outer_width=64, middle_width=64, inner_width=96),
    )
    state = TrainingState.fresh(config, device, "test-sha")
    obj = serialize_checkpoint(_runtime(state), config, code_sha="test-sha")
    ckpt = tmp_path / "epoch_000.pt"
    torch.save(obj, ckpt)

    loaded = load_checkpoint(ckpt, device)
    assert loaded.config == config
    assert loaded.cfg == config.arch
    src_sd, loaded_sd = state.model.state_dict(), loaded.model.state_dict()
    assert src_sd.keys() == loaded_sd.keys()
    for k in src_sd:
        assert torch.equal(src_sd[k], loaded_sd[k].cpu()), f"param mismatch: {k}"


# TODO(ckpt-cfg-refactor-note): old docstring described the v1 layout in detail.
# It should just be documented in one location that we can point to from here.
def test_v1_serialized_shape(tmp_path):
    """serialize_checkpoint emits the v1 top-level layout."""
    config = _v1_config(tmp_path)
    state = TrainingState.fresh(config, torch.device("cpu"), "test-sha")
    obj = serialize_checkpoint(_runtime(state), config, code_sha="abc123")

    assert set(obj) == {"model", "optim", "scaler", "epoch", "in_ch", "config", "code_sha"}
    assert obj["config"]["config_version"] == CONFIG_VERSION
    assert obj["in_ch"] == config.arch.in_ch
    assert obj["code_sha"] == "abc123"
    # paths should be stringified
    assert isinstance(obj["config"]["run_dir"], str)


def test_v1_in_ch_mismatch_raises(tmp_path):
    """A v1 checkpoint whose recorded in_ch contradicts its arch's obs channel
    count raises a clear error on load."""
    config = _v1_config(tmp_path)
    state = TrainingState.fresh(config, torch.device("cpu"), "test-sha")
    obj = serialize_checkpoint(_runtime(state), config, code_sha="x")
    obj["in_ch"] += 1
    ckpt = tmp_path / "epoch_000.pt"
    torch.save(obj, ckpt)

    with pytest.raises(ValueError, match="contradicts obs"):
        load_checkpoint(ckpt, torch.device("cpu"))


def test_is_arch_bearing_across_formats(tmp_path):
    """is_arch_bearing across all three layouts: v0 combined (top-level arch) and
    v1 (arch nested in config) are arch-bearing; a v0 bare state_dict is not."""
    device = torch.device("cpu")
    config = _v1_config(tmp_path)
    state = TrainingState.fresh(config, device, "test-sha")

    v0_combined = tmp_path / "v0_combined.pt"
    torch.save(
        {"model": state.model.state_dict(), "arch": asdict(config.arch), "epoch": 1},
        v0_combined,
    )
    v0_bare = tmp_path / "v0_bare.pt"
    torch.save(state.model.state_dict(), v0_bare)
    v1 = tmp_path / "v1.pt"
    torch.save(serialize_checkpoint(_runtime(state), config, code_sha="x"), v1)

    assert is_arch_bearing(v0_combined) is True
    assert is_arch_bearing(v0_bare) is False
    assert is_arch_bearing(v1) is True


def test_v1_model_key_ignores_legacy_variant_kwarg(tmp_path):
    """The value_head_variant arg only applies to legacy checkpoints — a v1 load
    ignores it, so it must not appear in model_key either: the same v1 file
    loaded under different variant args is one model and gets one key."""
    device = torch.device("cpu")
    config = _v1_config(tmp_path)
    state = TrainingState.fresh(config, device, "test-sha")
    ckpt = tmp_path / "epoch_000.pt"
    torch.save(serialize_checkpoint(_runtime(state), config, code_sha="x"), ckpt)

    direct = BCModelHandle.load(ckpt, device, value_head_variant="direct")
    pyramid = BCModelHandle.load(ckpt, device, value_head_variant="pyramid")
    assert direct.model_key == pyramid.model_key
