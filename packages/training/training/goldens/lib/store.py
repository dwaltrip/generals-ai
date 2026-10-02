"""The only module that reads or writes the references tree."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from training.goldens.lib.content import ArrayLayout, EntryContent, RowHashes, check_content
from training.goldens.lib.entry import EntryId


_SUFFIX = ".npz"
_META = "__meta__"


# --- Paths ---


def rel_path(eid: EntryId) -> Path:
    return Path(eid.surface) / eid.point / f"{eid.fixture}{_SUFFIX}"


def parse_rel_path(rel: Path) -> EntryId | None:
    parts = rel.parts
    if len(parts) != 3 or not parts[2].endswith(_SUFFIX):
        return None
    surface, point, name = parts
    fixture = name.removesuffix(_SUFFIX)
    if not fixture:
        return None
    return EntryId(surface=surface, point=point, fixture=fixture)


# Build a case-insensitive key for a path.
def path_key(eid: EntryId) -> str:
    return rel_path(eid).as_posix().casefold()


# Relative to root, sorted. Files with a dotted part anywhere in their path are skipped.
def _files_under(root: Path) -> list[Path]:
    rels = (p.relative_to(root) for p in root.rglob("*") if p.is_file())
    return sorted(
        (rel for rel in rels if not any(part.startswith(".") for part in rel.parts)),
        key=Path.as_posix,
    )


# --- Reading the tree ---


# The paths are relative to root.
@dataclass(frozen=True)
class ReferenceTree:
    root: Path
    references: dict[EntryId, EntryContent]   # in scan order
    unrecognized: list[Path]   # .npz files that don't parse, or are under an unknown surface
    ignored: list[Path]        # all other files

    def get_reference(self, eid: EntryId) -> EntryContent | None:
        return self.references.get(eid)


def load_references(root: Path, surfaces: Collection[str]) -> ReferenceTree:
    references: dict[EntryId, EntryContent] = {}
    unrecognized, ignored = [], []

    for rel in _files_under(root):
        if rel.suffix != _SUFFIX:
            ignored.append(rel)
        elif (eid := parse_rel_path(rel)) is None or eid.surface not in surfaces:
            unrecognized.append(rel)
        else:
            references[eid] = _load_file(root / rel)

    return ReferenceTree(
        root=root, references=references, unrecognized=unrecognized, ignored=ignored
    )


# --- Read / write ---


# Array names are sorted and arrays converted to C order, so that equal content
# always serializes to the same bytes. A RowHashes is stored as its hashes, with
# its layout in the `__meta__` member.
def serialize(content: EntryContent) -> bytes:
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}

    for name, value in content.items():
        if isinstance(value, RowHashes):
            layout = value.layout
            meta[name] = {
                "shape": list(layout.shape),
                "dtype": layout.dtype.str,
                "hashed_axis": layout.hashed_axis,
            }
            arrays[name] = value.hashes
        else:
            arrays[name] = value

    members: dict[str, Any] = {name: np.ascontiguousarray(arr) for name, arr in arrays.items()}
    if meta:
        members[_META] = np.array(json.dumps(meta, sort_keys=True))

    buf = io.BytesIO()
    np.savez(buf, **{name: members[name] for name in sorted(members)})
    return buf.getvalue()


def _load_file(path: Path) -> EntryContent:
    with np.load(path, allow_pickle=False) as z:
        members = {name: z[name] for name in z.files}

    try:
        return _content_from_members(members)
    except ValueError as e:
        e.add_note(f"in {path}")
        raise


# The inverse of the member building in `serialize`.
def _content_from_members(members: dict[str, np.ndarray]) -> EntryContent:
    meta = json.loads(str(members.pop(_META))) if _META in members else {}
    if not isinstance(meta, dict):
        raise ValueError(f"{_META} is not a JSON object: {meta!r}")
    content: EntryContent = dict(members)

    for name, m in meta.items():
        if name not in members:
            raise ValueError(f"{_META} lists {name!r}, which isn't in the file")
        try:
            layout = ArrayLayout(
                shape=tuple(m["shape"]), dtype=np.dtype(m["dtype"]), hashed_axis=m["hashed_axis"]
            )
        except (KeyError, TypeError) as e:
            raise ValueError(f"malformed {_META} entry for {name!r}: {m!r}") from e
        content[name] = RowHashes(hashes=members[name], layout=layout)

    check_content(content)
    return content


def save(eid: EntryId, content: EntryContent, root: Path) -> None:
    path = root / rel_path(eid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialize(content))


def remove(eid: EntryId, root: Path) -> None:
    (root / rel_path(eid)).unlink()


# --- Set digest ---


# sha256 over every .npz file under `root` in sorted path order, each framed as
# (relative path, bytes) with lengths, truncated to 16 hex characters.
# TODO: This includes the files `load_references` lists as unrecognized, such as those under
# a surface that is no longer registered. Revisit whether they belong in the
# digest when the revisions are designed.
def set_digest(root: Path) -> str:
    h = hashlib.sha256()
    for rel in _files_under(root):
        if rel.suffix != _SUFFIX:
            continue
        name = rel.as_posix().encode()
        data = (root / rel).read_bytes()
        h.update(len(name).to_bytes(4, "little"))
        h.update(name)
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()[:16]
