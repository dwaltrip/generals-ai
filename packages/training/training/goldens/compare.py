from __future__ import annotations

from collections.abc import Mapping
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


def to_stored(raw: dict[str, np.ndarray], refs: Mapping[str, KeyRef]) -> dict[str, np.ndarray]:
    return {key: stored_form(refs[key].form, arr) for key, arr in raw.items()}


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
    rows = np.nonzero((got != ref).reshape(got.shape[0], -1).any(axis=1))[0]
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
    diffs: tuple[KeyDiff, ...]
    note: str | None = None

    def summary(self) -> str:
        lines = [f"supervision mismatch: keys {[d.key for d in self.diffs]}"]
        lines += [f"  {d.summary()}" for d in self.diffs]
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)


def compare_supervision(
    got: dict[str, np.ndarray],
    ref: dict[str, np.ndarray],
    keys: tuple[str, ...],
) -> SupervisionMismatch | None:
    # `got` is in stored form (see `to_stored`).
    if set(got) != set(keys):
        extra = set(got) - set(keys)
        missing = set(keys) - set(got)
        return SupervisionMismatch(
            diffs=(),
            note=f"emitted keys differ. extra: {extra}, missing: {missing}",
        )
    diffs = tuple(d for key in keys if (d := diff_rows(key, got[key], ref[key])) is not None)
    return SupervisionMismatch(diffs=diffs) if diffs else None
