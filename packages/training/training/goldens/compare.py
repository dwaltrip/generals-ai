from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training.bc.datapipe.walk import WalkFrame
from training.goldens.hashes import hash_channels, hash_frames, hash_mask_frames
from training.goldens.loaders import ObsReference


# TODO: figure out dtype declarations for the goldens
# TODO: partial-bundle handling (key missing from the bundle file)


@dataclass(frozen=True)
class ObsMismatch:
    changed_ticks: np.ndarray
    changed_channels: np.ndarray
    note: str | None = None

    def summary(self) -> str:
        if self.note is not None:
            return f"obs mismatch: {self.note}"
        first = int(self.changed_ticks[0]) if self.changed_ticks.size else -1
        return (
            f"obs mismatch: {self.changed_ticks.size} ticks changed"
            f" (first t={first}), channels={self.changed_channels.tolist()}"
        )


@dataclass(frozen=True)
class SupervisionMismatch:
    changed_keys: tuple[str, ...]

    def summary(self) -> str:
        return f"supervision mismatch: keys {list(self.changed_keys)}"


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


def compare_supervision(
    got: dict[str, np.ndarray],
    bundle: dict[str, np.ndarray],
    keys: tuple[str, ...],
) -> SupervisionMismatch | None:
    changed = []
    for key in keys:
        if key == "legality_mask":
            same = np.array_equal(
                hash_mask_frames(got["legality_mask"]),
                bundle["legality_mask_frame_hashes"],
            )
        else:
            ref = bundle[key]
            same = (
                got[key].dtype == ref.dtype
                and got[key].shape == ref.shape
                and got[key].tobytes() == ref.tobytes()
            )
        if not same:
            changed.append(key)
    return SupervisionMismatch(changed_keys=tuple(changed)) if changed else None
