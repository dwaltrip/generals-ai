from dataclasses import dataclass
from typing import Any, cast

import torch

from training.bc.datapipe.emit import FrameEmission
from training.bc.datapipe.sim_types import SimFrame
from training.bc.datapipe.walk import WalkFrame


@dataclass(frozen=True)
class FrameMeta:
    frame_t: torch.Tensor
    players_alive: torch.Tensor
    p_start: torch.Tensor
    sample_idx: torch.Tensor

    def shallow_dict(self) -> dict[str, torch.Tensor]:
        return dict(vars(self))


@dataclass(frozen=True)
class TrainingSample:
    # core fields
    obs: torch.Tensor
    mask: torch.Tensor
    valid_mask: torch.Tensor
    action_target: torch.Tensor
    is_pass: torch.Tensor
    value_target: torch.Tensor

    frame_meta: FrameMeta | None
    sim_frame: SimFrame | None

    alive_mask: torch.Tensor | None
    aux_head_targets: dict[str, torch.Tensor] | None

    def to_dict(self) -> dict[str, torch.Tensor]:
        sample = {
            "obs": self.obs,
            "mask": self.mask,
            "valid_mask": self.valid_mask,
            "action_target": self.action_target,
            "is_pass": self.is_pass,
            "value_target": self.value_target,
        }

        if self.frame_meta is not None:
            # Don't use dataclasses.asdict, as that deep copies the values.
            sample.update(self.frame_meta.shallow_dict())

        if self.alive_mask is not None:
            sample["alive_mask"] = self.alive_mask

        if self.aux_head_targets is not None:
            sample.update(self.aux_head_targets)

        if self.sim_frame is not None:
            # Analysis-only seam, deliberately non-Tensor.
            # This path should never run during an actual train loop.
            # See `EmitSpec.attach_sim_frame`.
            sample["sim_frame"] = cast(Any, self.sim_frame)

        return sample


def pack_sample(
    frame: WalkFrame,
    emission: FrameEmission,
    frame_meta: FrameMeta | None,
    sim_frame: SimFrame | None,
) -> TrainingSample:
    return TrainingSample(
        obs=torch.from_numpy(frame.obs),
        mask=torch.from_numpy(emission.legality_mask),
        valid_mask=torch.from_numpy(frame.board_mask),
        action_target=torch.from_numpy(emission.action_target),
        is_pass=torch.from_numpy(emission.is_pass),
        value_target=torch.from_numpy(emission.value_target),
        frame_meta=frame_meta,
        sim_frame=sim_frame,
        alive_mask=(
            None if emission.alive_mask is None else torch.from_numpy(emission.alive_mask)
        ),
        aux_head_targets=(
            None
            if emission.aux_head_targets is None
            else {
                key: torch.from_numpy(value)
                for key, value in emission.aux_head_targets.items()
            }
        ),
    )
