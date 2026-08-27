from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from training.bc.datapipe.emit_spec import PartialEmitSpec
from training.bc.obs_config import ObsConfig


# TODO(sketch): registry data transcription pending (8.12-2), including the
# emit-point naming decision (targets vs emit vocabulary).


@dataclass(frozen=True)
class FixtureRecord:
    replay_id: str
    slot: int
    note: str

    @property
    def id(self) -> str:
        return f"{self.replay_id}-s{self.slot}"


@dataclass(frozen=True)
class ObsEntry:
    point: str
    cfg: ObsConfig
    fixture: FixtureRecord
    ref_path: Path


@dataclass(frozen=True)
class TargetsEntry:
    point: str
    spec: PartialEmitSpec
    keys: tuple[str, ...]
    fixture: FixtureRecord
    bundle_path: Path


def obs_entries() -> list[ObsEntry]:
    # TODO(sketch)
    return []


def targets_entries() -> list[TargetsEntry]:
    # TODO(sketch)
    return []
