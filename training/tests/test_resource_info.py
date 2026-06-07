"""Smoke coverage for the startup resource logger."""

from __future__ import annotations

import torch

from shared.resource_info import log_resource_info


def test_log_resource_info_smoke(capsys):
    # CPU device exercises the Linux-guarded branches without needing CUDA.
    log_resource_info(
        device=torch.device("cpu"),
        batch_size=512,
        obs_channels=126,
        obs_dtype="fp32",
        spatial=(32, 32),
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
    )
    out = capsys.readouterr().out
    assert "pipeline host-memory footprint" in out
    assert "per-batch obs: 252.0 MiB" in out
    assert "× fp32)" in out
    assert "16 batches" in out


def test_log_resource_info_fp16_footprint(capsys):
    # fp16 obs halves the per-batch footprint and is labelled as such.
    log_resource_info(
        device=torch.device("cpu"),
        batch_size=512,
        obs_channels=126,
        obs_dtype="fp16",
        spatial=(32, 32),
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
    )
    out = capsys.readouterr().out
    assert "per-batch obs: 126.0 MiB" in out
    assert "× fp16)" in out
