from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from training.goldens.hashes import ObsDigest, hash_along_first_axis
from training.goldens.registry import KeyRef, RefForm


def same_bytes(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _changed_indices_1d(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    assert a.ndim == 1 and b.ndim == 1
    if a.size == b.size:
        return np.nonzero(a != b)[0]
    return None


def stored_form(form: RefForm, arr: np.ndarray) -> np.ndarray:
    match form:
        case RefForm.FULL:
            return arr
        case RefForm.FRAME_HASHES:
            return hash_along_first_axis(arr)


@dataclass(frozen=True)
class KeysetDiff:
    extra: frozenset[str]
    missing: frozenset[str]


def keyset_diff(emitted: Iterable[str], declared: Iterable[str]) -> KeysetDiff | None:
    emitted, declared = set(emitted), set(declared)
    if emitted == declared:
        return None
    return KeysetDiff(
        extra=frozenset(emitted - declared),
        missing=frozenset(declared - emitted),
    )


@dataclass(frozen=True)
class ArrayLayout:
    shape: tuple[int, ...]
    dtype: np.dtype

    @classmethod
    def of(cls, arr: np.ndarray) -> ArrayLayout:
        return cls(shape=arr.shape, dtype=arr.dtype)


@dataclass(frozen=True)
class KeyDiff:
    key: str
    ref: ArrayLayout
    got: ArrayLayout
    changed_rows: np.ndarray | None   # indices along the first axis. None when the layouts differ.


def diff_rows(key: str, got: np.ndarray, ref: np.ndarray) -> KeyDiff | None:
    ref_layout, got_layout = ArrayLayout.of(ref), ArrayLayout.of(got)
    if ref_layout != got_layout:
        return KeyDiff(key=key, ref=ref_layout, got=got_layout, changed_rows=None)
    # Rows are compared by their bytes, matching `same_bytes`. Comparing values
    # would differ for floats: NaN never equals itself, and -0.0 equals 0.0.
    got_b = np.ascontiguousarray(got).view(np.uint8).reshape(got.shape[0], -1)
    ref_b = np.ascontiguousarray(ref).view(np.uint8).reshape(ref.shape[0], -1)
    rows = np.nonzero((got_b != ref_b).any(axis=1))[0]
    if rows.size == 0:
        return None
    return KeyDiff(key=key, ref=ref_layout, got=got_layout, changed_rows=rows)


# --- Obs ---


@dataclass(frozen=True)
class ObsShape:
    n_frames: int
    n_channels: int

    @classmethod
    def of(cls, digest: ObsDigest) -> ObsShape:
        return cls(n_frames=digest.frame_hashes.size, n_channels=digest.channel_hashes.size)


@dataclass(frozen=True)
class ObsMismatch:
    ref: ObsShape
    got: ObsShape
    changed_frames: np.ndarray | None     # None when the frame counts differ
    changed_channels: np.ndarray | None   # None when the channel counts differ


# Frames and channels are compared independently.
# Diffs are only computed when the arrays have matching size.
def compare_obs(got: ObsDigest, ref: ObsDigest) -> ObsMismatch | None:
    frames = _changed_indices_1d(got.frame_hashes, ref.frame_hashes)
    channels = _changed_indices_1d(got.channel_hashes, ref.channel_hashes)
    if frames is not None and channels is not None and frames.size == 0 and channels.size == 0:
        return None
    return ObsMismatch(
        ref=ObsShape.of(ref),
        got=ObsShape.of(got),
        changed_frames=frames,
        changed_channels=channels,
    )


# --- Supervision ---


@dataclass(frozen=True)
class SupervisionMismatch:
    keyset: KeysetDiff | None = None
    diffs: tuple[KeyDiff, ...] = ()


def compare_supervision(
    raw: dict[str, np.ndarray],
    ref: dict[str, np.ndarray],
    refs: Mapping[str, KeyRef],
) -> SupervisionMismatch | None:
    if (kd := keyset_diff(raw, refs)) is not None:
        return SupervisionMismatch(keyset=kd)
    diffs = tuple(
        d
        for key, r in refs.items()
        if (d := diff_rows(key, stored_form(r.form, raw[key]), ref[key])) is not None
    )
    return SupervisionMismatch(diffs=diffs) if diffs else None
