"""Smoke coverage for the startup resource logger."""

from __future__ import annotations

import torch

from shared.resource_info import _fmt_bytes, log_resource_info


def test_fmt_bytes():
    assert _fmt_bytes(0) == "0.0 B"
    assert _fmt_bytes(1024) == "1.0 KiB"
    assert _fmt_bytes(1024**3) == "1.0 GiB"
    # The per-batch obs figure at n=20 (bs=512, 126 ch, 32x32, fp32).
    assert _fmt_bytes(512 * 126 * 32 * 32 * 4) == "252.0 MiB"


def test_log_resource_info_smoke(capsys):
    # CPU device exercises the Linux-guarded branches without needing CUDA.
    log_resource_info(
        device=torch.device("cpu"),
        batch_size=512,
        obs_channels=126,
        spatial=(32, 32),
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
    )
    out = capsys.readouterr().out
    assert "pipeline host-memory footprint" in out
    assert "per-batch obs: 252.0 MiB" in out
    assert "16 batches" in out
