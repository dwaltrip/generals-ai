"""
The on-disk layout of the references tree, and reading and writing it.

    obs/<point>/<fixture>.npz
    supervision/<fixture>/<point>@<key>.npy

For supervision, <point> is the representative point for the key (see
registry.representative). Nothing here imports the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path

import numpy as np

from training.goldens.hashes import ObsDigest
from training.goldens.paths import REFERENCES_DIR


REP_SEP = "@"


class Surface(Enum):
    OBS = "obs"
    SUPERVISION = "supervision"


@dataclass(frozen=True)
class RefId:
    surface: Surface
    point: str
    fixture: str
    key: str | None = None  # supervision only

    def __post_init__(self) -> None:
        assert (self.key is None) == (self.surface is Surface.OBS), self


# --- Paths ---


def ref_path(rid: RefId) -> Path:
    match rid.surface:
        case Surface.OBS:
            return REFERENCES_DIR / "obs" / rid.point / f"{rid.fixture}.npz"
        case Surface.SUPERVISION:
            return (
                REFERENCES_DIR / "supervision" / rid.fixture / f"{rid.point}{REP_SEP}{rid.key}.npy"
            )


def parse_ref_path(path: Path) -> RefId | None:
    try:
        parts = path.relative_to(REFERENCES_DIR).parts
    except ValueError:
        return None
    if len(parts) != 3:
        return None
    surface, middle, name = parts
    if surface == "obs" and name.endswith(".npz"):
        return RefId(Surface.OBS, point=middle, fixture=name[: -len(".npz")])
    if surface == "supervision" and name.endswith(".npy") and REP_SEP in name:
        point, key = name[: -len(".npy")].split(REP_SEP, 1)
        return RefId(Surface.SUPERVISION, point=point, fixture=middle, key=key)
    return None


def rel(rid: RefId) -> str:
    return ref_path(rid).relative_to(REFERENCES_DIR).as_posix()


def _is_dotted(path: Path, root: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def _files_under(root: Path) -> list[Path]:
    files = (p for p in root.rglob("*") if p.is_file() and not _is_dotted(p, root))
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


@dataclass(frozen=True)
class Listing:
    refs: list[RefId]
    unrecognized: list[Path]  # files on the tree that don't parse


def list_refs() -> Listing:
    refs, unrecognized = [], []
    for path in _files_under(REFERENCES_DIR):
        rid = parse_ref_path(path)
        if rid is None:
            unrecognized.append(path)
        else:
            refs.append(rid)
    return Listing(refs=refs, unrecognized=unrecognized)


# --- Read / write ---


def load_obs(rid: RefId) -> ObsDigest | None:
    path = ref_path(rid)
    if not path.exists():
        return None
    with np.load(path) as z:
        return ObsDigest(frame_hashes=z["frame_hashes"], channel_hashes=z["channel_hashes"])


def save_obs(rid: RefId, digest: ObsDigest) -> None:
    path = ref_path(rid)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, frame_hashes=digest.frame_hashes, channel_hashes=digest.channel_hashes)


def load_supervision(rid: RefId) -> np.ndarray | None:
    path = ref_path(rid)
    return np.load(path) if path.exists() else None


def save_supervision(rid: RefId, arr: np.ndarray) -> None:
    path = ref_path(rid)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def remove(rid: RefId) -> None:
    ref_path(rid).unlink()


# --- Set digest ---


# sha256 over every file under `root` in sorted path order, each framed as
# (relative path, bytes) with lengths, truncated to 16 hex characters.
def set_digest(root: Path = REFERENCES_DIR) -> str:
    h = hashlib.sha256()
    for path in _files_under(root):
        name = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        h.update(len(name).to_bytes(4, "little"))
        h.update(name)
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()[:16]
