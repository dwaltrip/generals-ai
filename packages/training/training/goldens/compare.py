from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training.bc.datapipe.walk import WalkFrame
from training.goldens.hashes import hash_channels, hash_frames, hash_mask_frames
from training.goldens.loaders import ObsReference


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


@dataclass(frozen=True)
class SupervisionMismatch:
    changed_keys: tuple[str, ...]
    note: str | None = None

    def summary(self) -> str:
        note = f"\n  note: {self.note}" if self.note else ""
        return f"supervision mismatch: keys {list(self.changed_keys)}{note}"


def same_bytes(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def compare_obs(frames: list[WalkFrame], ref: ObsReference) -> ObsMismatch | None:
    frame_hashes = hash_frames(frames)
    channel_hashes = hash_channels(frames)
    if (
        frame_hashes.shape != ref.frame_hashes.shape
        or channel_hashes.shape != ref.channel_hashes.shape
    ):
        return ObsMismatch(
            changed_ticks=np.empty(0, dtype=np.int64),
            changed_channels=np.empty(0, dtype=np.int64),
            note=(
                f"shape mismatch: frames {frame_hashes.shape} vs ref"
                f" {ref.frame_hashes.shape}, channels {channel_hashes.shape}"
                f" vs ref {ref.channel_hashes.shape}"
            ),
        )
    changed_ticks = np.nonzero(frame_hashes != ref.frame_hashes)[0]
    changed_channels = np.nonzero(channel_hashes != ref.channel_hashes)[0]
    if changed_ticks.size == 0 and changed_channels.size == 0:
        return None
    return ObsMismatch(changed_ticks=changed_ticks, changed_channels=changed_channels)


def stored_form(key: str, arr: np.ndarray, hashed_keys: frozenset[str]) -> np.ndarray:
    return hash_mask_frames(arr) if key in hashed_keys else arr


def compare_supervision(
    got: dict[str, np.ndarray],
    ref: dict[str, np.ndarray],
    keys: tuple[str, ...],
    hashed_keys: frozenset[str],
) -> SupervisionMismatch | None:
    if set(got) != set(keys):
        extra = set(got) - set(keys)
        missing = set(got) - set(keys)
        return SupervisionMismatch(
            changed_keys=(),
            note=f"emitted keys differ. extra: {extra}, missing: {missing}",
        )
    changed = tuple(
        key for key in keys
        if not same_bytes(stored_form(key, got[key], hashed_keys), ref[key])
    )
    return SupervisionMismatch(changed_keys=changed) if changed else None
