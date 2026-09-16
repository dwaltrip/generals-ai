from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from training.goldens.hashes import ObsDigest, hash_along_first_axis
from training.goldens.registry import KeyRef, RefForm


def same_bytes(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# --- Stored form ---


def stored_form(form: RefForm, arr: np.ndarray) -> np.ndarray:
    match form:
        case RefForm.FULL:
            return arr
        case RefForm.FRAME_HASHES:
            return hash_along_first_axis(arr)


# --- Keyset diff, shared by the tests and regen ---


@dataclass(frozen=True)
class KeysetDiff:
    extra: frozenset[str]    # emitted but not declared
    missing: frozenset[str]  # declared but not emitted

    def summary(self) -> str:
        return f"emitted keys differ. extra: {set(self.extra)}, missing: {set(self.missing)}"


def keyset_diff(emitted: Iterable[str], declared: Iterable[str]) -> KeysetDiff | None:
    emitted, declared = set(emitted), set(declared)
    if emitted == declared:
        return None
    return KeysetDiff(extra=frozenset(emitted - declared), missing=frozenset(declared - emitted))


# --- Per-key row diff, shared by the tests and regen ---


@dataclass(frozen=True)
class KeyDiff:
    key: str
    changed_rows: np.ndarray   # indices along the first axis (ticks, for per-frame arrays)
    total_rows: int
    note: str | None = None    # set when shapes or dtypes differ; rows are then empty

    def summary(self) -> str:
        if self.note:
            return f"{self.key}: {self.note}"
        return (
            f"{self.key}: {self.changed_rows.size} of {self.total_rows} rows changed,"
            f" first t={int(self.changed_rows[0])}"
        )


def diff_rows(key: str, got: np.ndarray, ref: np.ndarray) -> KeyDiff | None:
    if got.shape != ref.shape or got.dtype != ref.dtype:
        return KeyDiff(
            key=key,
            changed_rows=np.empty(0, dtype=np.int64),
            total_rows=got.shape[0],
            note=f"shape/dtype {got.shape} {got.dtype} vs ref {ref.shape} {ref.dtype}",
        )
    # Rows are compared by their bytes, matching `same_bytes`. Comparing values
    # would differ for floats: NaN never equals itself, and -0.0 equals 0.0.
    got_b = np.ascontiguousarray(got).view(np.uint8).reshape(got.shape[0], -1)
    ref_b = np.ascontiguousarray(ref).view(np.uint8).reshape(ref.shape[0], -1)
    rows = np.nonzero((got_b != ref_b).any(axis=1))[0]
    if rows.size == 0:
        return None
    return KeyDiff(key=key, changed_rows=rows, total_rows=got.shape[0])


# --- Obs ---


@dataclass(frozen=True)
class ObsMismatch:
    changed_ticks: np.ndarray
    changed_channels: np.ndarray
    note: str | None = None

    def summary(self) -> str:
        note = f"\n  note: {self.note}" if self.note else ""
        first = int(self.changed_ticks[0]) if self.changed_ticks.size else -1
        return (
            f"obs mismatch: {self.changed_ticks.size} ticks changed" +
            f" (first t={first}), channels={self.changed_channels.tolist()}" +
            note
        )


def compare_obs(got: ObsDigest, ref: ObsDigest) -> ObsMismatch | None:
    if (
        got.frame_hashes.shape != ref.frame_hashes.shape
        or got.channel_hashes.shape != ref.channel_hashes.shape
    ):
        return ObsMismatch(
            changed_ticks=np.empty(0, dtype=np.int64),
            changed_channels=np.empty(0, dtype=np.int64),
            note=(
                f"shape mismatch: frames {got.frame_hashes.shape} vs ref"
                f" {ref.frame_hashes.shape}, channels {got.channel_hashes.shape}"
                f" vs ref {ref.channel_hashes.shape}"
            ),
        )
    changed_ticks = np.nonzero(got.frame_hashes != ref.frame_hashes)[0]
    changed_channels = np.nonzero(got.channel_hashes != ref.channel_hashes)[0]
    if changed_ticks.size == 0 and changed_channels.size == 0:
        return None
    return ObsMismatch(changed_ticks=changed_ticks, changed_channels=changed_channels)


# --- Supervision ---


@dataclass(frozen=True)
class SupervisionMismatch:
    keyset: KeysetDiff | None = None
    diffs: tuple[KeyDiff, ...] = ()

    def summary(self) -> str:
        if self.keyset:
            return f"supervision mismatch: {self.keyset.summary()}"
        lines = [f"supervision mismatch: keys {[d.key for d in self.diffs]}"]
        return "\n".join(lines + [f"  {d.summary()}" for d in self.diffs])


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
