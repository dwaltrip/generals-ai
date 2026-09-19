from pathlib import Path

import pytest

from training.goldens.paths import REFERENCES_DIR
from training.goldens.registry import obs_entries, supervision_entries
from training.goldens.store import RefId, Surface, parse_ref_path, ref_path


def _registry_refs() -> list[RefId]:
    refs = [e.ref for e in obs_entries()]
    refs += [r.ref for e in supervision_entries() for r in e.refs.values()]
    return refs


def _rid_label(rid: RefId) -> str:
    return f"{rid.surface.value}:{rid.point}:{rid.fixture}:{rid.key}"


@pytest.mark.parametrize("rid", _registry_refs(), ids=_rid_label)
def test_ref_path_round_trip(rid: RefId) -> None:
    assert parse_ref_path(ref_path(rid)) == rid


def test_parse_rejects_off_scheme_paths() -> None:
    assert parse_ref_path(Path("/elsewhere/obs/p/f.npz")) is None
    assert parse_ref_path(REFERENCES_DIR / "obs" / "f.npz") is None
    assert parse_ref_path(REFERENCES_DIR / "obs" / "p" / "f.txt") is None
    assert parse_ref_path(REFERENCES_DIR / "supervision" / "f" / "no_separator.npy") is None
    assert parse_ref_path(REFERENCES_DIR / "other" / "p" / "f.npz") is None


def test_ref_id_key_matches_surface() -> None:
    with pytest.raises(AssertionError):
        RefId(Surface.OBS, point="p", fixture="f", key="k")
    with pytest.raises(AssertionError):
        RefId(Surface.SUPERVISION, point="p", fixture="f")
