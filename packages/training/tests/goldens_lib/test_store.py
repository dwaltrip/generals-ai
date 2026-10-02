import json
from pathlib import Path

import numpy as np
import pytest

from training.goldens.lib import store
from training.goldens.lib.content import EntryContent
from training.goldens.lib.diff import diff_content
from training.goldens.lib.entry import EntryId
from training.goldens.lib.hashing import row_hashes


_EID = EntryId(surface="obs", point="fp16", fixture="abc-s1")


def _content() -> EntryContent:
    raw = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    return {
        "full": np.arange(6, dtype=np.int64).reshape(3, 2),
        "hashed": row_hashes(raw, axis=1),
    }


def test_paths() -> None:
    assert store.rel_path(_EID) == Path("obs/fp16/abc-s1.npz")
    assert store.parse_rel_path(store.rel_path(_EID)) == _EID
    for off_scheme in ["obs/abc.npz", "a/b/c/d.npz", "obs/fp16/abc.npy", "obs/fp16/.npz"]:
        assert store.parse_rel_path(Path(off_scheme)) is None, off_scheme


def test_save_load_round_trip(tmp_path: Path) -> None:
    assert store.load_references(tmp_path, {"obs"}).get_reference(_EID) is None
    content = _content()
    store.save(_EID, content, tmp_path)
    loaded = store.load_references(tmp_path, {"obs"}).get_reference(_EID)
    assert loaded is not None
    assert diff_content(loaded, content) is None   # includes the RowHashes layout


def test_serialize_ignores_dict_and_memory_order() -> None:
    content = _content()
    reordered: EntryContent = {
        "hashed": content["hashed"],
        "full": np.asfortranarray(content["full"]),
    }
    assert store.serialize(reordered) == store.serialize(content)


@pytest.mark.parametrize(
    "meta",
    [
        # Lists an array that isn't in the file.
        {"other": {"shape": [3, 2], "dtype": "<f2", "hashed_axis": 0}},
        # Three hashes stored, but the layout implies four.
        {"a": {"shape": [4, 2], "dtype": "<f2", "hashed_axis": 0}},
        # Missing a key, or not an object.
        {"a": {"shape": [3]}},
        {"a": [3]},
    ],
)
def test_malformed_meta_rejected(tmp_path: Path, meta: dict) -> None:
    path = tmp_path / store.rel_path(_EID)
    path.parent.mkdir(parents=True)
    np.savez(path, a=np.arange(3, dtype=np.uint64), __meta__=np.array(json.dumps(meta)))
    with pytest.raises(ValueError):
        store.load_references(tmp_path, {"obs"})
