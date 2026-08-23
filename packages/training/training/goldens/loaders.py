from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.bc.sim_types import PerspectiveMeta
from training.goldens.registry import FixtureRecord


@dataclass(frozen=True)
class ObsReference:
    frame_hashes: np.ndarray
    channel_hashes: np.ndarray


def load_fixture(
    fixture: FixtureRecord,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    # TODO(sketch)
    raise NotImplementedError


def perspective_for(meta: dict[str, np.ndarray], slot: int) -> PerspectiveMeta:
    # TODO(sketch): slot -> k resolution
    raise NotImplementedError


def load_obs_reference(path: Path) -> ObsReference | None:
    # TODO(sketch): None = unblessed
    raise NotImplementedError


def load_targets_bundle(path: Path) -> dict[str, np.ndarray] | None:
    # TODO(sketch): None = unblessed
    raise NotImplementedError
