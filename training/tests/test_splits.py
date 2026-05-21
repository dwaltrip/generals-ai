"""
Smoke tests for `bc.splits` — the manifest producer + consumer.

Determinism and disjointness are the load-bearing claims of the module
(seeded shuffle = reproducible run). Round-trip through `samples_for_split`
covers the path-resolution seam that the dataset depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

from bc.splits import (
    MANIFEST_VERSION,
    build_manifest,
    load_manifest,
    samples_for_split,
)


def test_same_seed_is_deterministic(intermediate_root: Path, sim_paths: list[Path]) -> None:
    """Two builds with the same seed produce byte-identical train/val lists."""
    sub = sim_paths[:100]
    m1 = build_manifest(intermediate_root, seed=42, sim_paths=sub)
    m2 = build_manifest(intermediate_root, seed=42, sim_paths=sub)
    assert m1["train"] == m2["train"]
    assert m1["val"] == m2["val"]


def test_different_seed_differs(intermediate_root: Path, sim_paths: list[Path]) -> None:
    """Different seeds reshuffle the val draw."""
    sub = sim_paths[:100]
    m1 = build_manifest(intermediate_root, seed=42, sim_paths=sub)
    m2 = build_manifest(intermediate_root, seed=43, sim_paths=sub)
    assert m1["val"] != m2["val"]


def test_train_val_disjoint_and_cover_kept(
    intermediate_root: Path, sim_paths: list[Path]
) -> None:
    sub = sim_paths[:100]
    m = build_manifest(intermediate_root, seed=0, sim_paths=sub, val_frac=0.05)
    train = {tuple(p) for p in m["train"]}
    val = {tuple(p) for p in m["val"]}
    assert train.isdisjoint(val)
    assert len(train) + len(val) == m["kept_pairs"]
    # 5% of N rounded — within ±1 of expected for tiny N
    expected_val = round(m["kept_pairs"] * 0.05)
    assert abs(len(val) - expected_val) <= 1


def test_manifest_provenance_fields_present(
    intermediate_root: Path, sim_paths: list[Path]
) -> None:
    """The fields a training script needs to detect 'corpus moved under me' must exist."""
    m = build_manifest(intermediate_root, seed=0, sim_paths=sim_paths[:50])
    for key in (
        "version", "seed", "built_at", "filter_version", "git_sha",
        "corpus_size", "dropped_games", "kept_pairs", "val_frac",
        "train", "val",
    ):
        assert key in m, f"manifest missing provenance field: {key}"
    assert m["version"] == MANIFEST_VERSION


def test_load_and_resolve_roundtrip(
    intermediate_root: Path, sim_paths: list[Path], tmp_path: Path
) -> None:
    """Build → write → load → samples_for_split yields valid (Path, int) tuples."""
    m = build_manifest(intermediate_root, seed=7, sim_paths=sim_paths[:50])
    out_path = tmp_path / "v1.json"
    with out_path.open("w") as fp:
        json.dump(m, fp)

    loaded = load_manifest(out_path)
    train_samples = samples_for_split(loaded, "train", intermediate_root)
    val_samples = samples_for_split(loaded, "val", intermediate_root)

    assert len(train_samples) == len(m["train"])
    assert len(val_samples) == len(m["val"])

    # Resolved paths should exist on disk — sim_path_for must match the parser's layout.
    for sim_path, k in train_samples[:5] + val_samples[:5]:
        assert sim_path.exists(), f"resolved path doesn't exist: {sim_path}"
        assert isinstance(k, int)
