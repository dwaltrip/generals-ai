"""
Pins the goldens machinery itself against synthetic input: the obs hashing,
the stored forms, and the set digest. No fixture or guarded code is involved.

If the golden tests fire and this test fails too, the goldens' own mechanism
changed. Re-pin the literals in the same commit and record the revision as
mechanism-change (9.18-1 section 11).
"""

from pathlib import Path

import numpy as np

from training.goldens.compare import stored_form
from training.goldens.hashes import ObsHasher
from training.goldens.registry import RefForm
from training.goldens.store import set_digest


def _frames() -> list[np.ndarray]:
    base = np.arange(12, dtype=np.float16).reshape(3, 2, 2)
    return [base * (i + 1) for i in range(2)]


def test_obs_hasher() -> None:
    hasher = ObsHasher()
    for frame in _frames():
        hasher.add(frame)
    digest = hasher.digest()
    assert digest.frame_hashes.dtype == np.uint64
    assert digest.channel_hashes.dtype == np.uint64
    assert digest.frame_hashes.tolist() == [7011111607144071736, 5775442798051356653]
    assert digest.channel_hashes.tolist() == [
        7547740460534904907, 17114147300992841919, 10854986861722603265,
    ]


def test_stored_forms() -> None:
    arr = np.arange(6, dtype=np.int64).reshape(3, 2)
    assert stored_form(RefForm.FULL, arr) is arr
    hashed = stored_form(RefForm.FRAME_HASHES, arr)
    assert hashed.dtype == np.uint64
    assert hashed.tolist() == [8639909309411767453, 9977687269915323902, 12965741279987845181]


def test_set_digest(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.npy").write_bytes(b"xyz")
    (tmp_path / "b.txt").write_bytes(b"hello")
    assert set_digest(tmp_path) == "a9fe98c8383528cb"

    # Dotfiles and dot-directories are skipped.
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "c").write_bytes(b"junk")
    assert set_digest(tmp_path) == "a9fe98c8383528cb"

    # The path is part of the digest, so a moved file changes it.
    (tmp_path / "b.txt").rename(tmp_path / "a" / "b.txt")
    assert set_digest(tmp_path) != "a9fe98c8383528cb"
