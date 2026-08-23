from __future__ import annotations

import numpy as np

from training.bc.walk import WalkFrame

# TODO(sketch): hash algorithm + canonical serialization decided in the
# hash-builder sketch, recorded in the channel summary.


def hash_frames(frames: list[WalkFrame]) -> np.ndarray:
    raise NotImplementedError


def hash_channels(frames: list[WalkFrame]) -> np.ndarray:
    raise NotImplementedError


def hash_mask_frames(masks: np.ndarray) -> np.ndarray:
    raise NotImplementedError
