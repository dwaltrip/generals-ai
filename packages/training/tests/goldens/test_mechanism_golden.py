"""
Pins the goldens machinery itself against synthetic input: the row hashing,
serialization, and the set digest. No fixture or guarded code is involved.

This test fails if the behavior of the core machinery has changed. This is either
a bug or an intentional change. Intentional changes require a revision record with
"mechanism-change" as the "kind". The hard-coded values below must also be updated.
"""

import hashlib
from pathlib import Path

import numpy as np

from training.goldens.lib import EntryContent, StreamingRowHasher, row_hashes, store


# TODO: Pull the hard-code values below out into module constants (or something).

def _frames() -> list[np.ndarray]:
    base = np.arange(12, dtype=np.float16).reshape(3, 2, 2)
    return [base * (i + 1) for i in range(2)]


def test_streaming_row_hasher() -> None:
    hasher = StreamingRowHasher(axes=(0, 1))
    for frame in _frames():
        hasher.add(frame)
    frame_hashes, channel_hashes = hasher.hashes()
    assert frame_hashes.hashes.tolist() == [7011111607144071736, 5775442798051356653]
    assert channel_hashes.hashes.tolist() == [
        7547740460534904907, 17114147300992841919, 10854986861722603265,
    ]


def test_row_hashes() -> None:
    hashed = row_hashes(np.arange(6, dtype=np.int64).reshape(3, 2))
    assert hashed.hashes.tolist() == [
        8639909309411767453, 9977687269915323902, 12965741279987845181,
    ]


# A format change, from our code or from a numpy or Python upgrade, changes the bytes.
def test_serialize() -> None:
    content: EntryContent = {
        "full": np.arange(6, dtype=np.int16).reshape(3, 2),
        "hashed": row_hashes(np.arange(6, dtype=np.float16).reshape(3, 2)),
    }
    assert hashlib.sha256(store.serialize(content)).hexdigest()[:16] == "5a21ba165ee77e3f"


def test_set_digest(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.npy").write_bytes(b"xyz")
    (tmp_path / "b.npz").write_bytes(b"hello")
    assert store.set_digest(tmp_path) == "baccad2d2fe535fa"

    # Dotfiles, dot-directories, and non-reference files are outside the digest.
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "c").write_bytes(b"junk")
    (tmp_path / "README.md").write_bytes(b"notes")
    assert store.set_digest(tmp_path) == "baccad2d2fe535fa"

    # The path is part of the digest, so a moved file changes it.
    (tmp_path / "b.npz").rename(tmp_path / "a" / "b.npz")
    assert store.set_digest(tmp_path) != "baccad2d2fe535fa"
