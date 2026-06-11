"""
Smoke tests for `analysis.run_metrics` — the sidecar staleness contract.

A sidecar is valid iff its `manifest_sha256` matches the manifest's current
content; anything else (missing field, mismatched hash) counts as missing.
Computing real marginals needs the corpus, so the recompute path is stubbed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.analysis import run_metrics
from training.analysis.run_metrics import get_manifest_metrics, sidecar_path


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    p = tmp_path / "split.json"
    p.write_text('{"train": [], "val": []}')
    return p


def _write_sidecar(manifest: Path, content: dict) -> None:
    sidecar_path(manifest).write_text(json.dumps(content))


def _valid_sidecar(manifest: Path) -> dict:
    return {
        "manifest": manifest.name,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "val_marginals": {"H_frame": 1.5},
    }


def test_matching_hash_returns_content(manifest: Path) -> None:
    _write_sidecar(manifest, _valid_sidecar(manifest))
    metrics = get_manifest_metrics(manifest)
    assert metrics is not None
    assert metrics["val_marginals"]["H_frame"] == 1.5


def test_stale_or_absent_returns_none_without_create(manifest: Path) -> None:
    assert get_manifest_metrics(manifest) is None  # no sidecar at all

    stale = _valid_sidecar(manifest)
    del stale["manifest_sha256"]  # pre-hash sidecar
    _write_sidecar(manifest, stale)
    assert get_manifest_metrics(manifest) is None

    _write_sidecar(manifest, _valid_sidecar(manifest))
    manifest.write_text('{"train": [], "val": ["changed"]}')  # --force rebuild
    assert get_manifest_metrics(manifest) is None


def test_stale_triggers_recompute_with_create(
    manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = _valid_sidecar(manifest)
    stale["manifest_sha256"] = "0" * 64
    _write_sidecar(manifest, stale)

    fresh = _valid_sidecar(manifest)
    monkeypatch.setattr(
        run_metrics, "compute_manifest_metrics", lambda path, intermediate: fresh
    )
    metrics = get_manifest_metrics(manifest, create=True)
    assert metrics is fresh
    # The backfill is persisted, so the next read needs no recompute.
    assert get_manifest_metrics(manifest) == fresh
