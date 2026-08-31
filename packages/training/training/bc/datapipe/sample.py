from dataclasses import dataclass
from typing import Any, cast

import torch

from training.bc.datapipe.emit import FrameSupervision
from training.bc.datapipe.sim_types import SimFrame
from training.bc.datapipe.walk import WalkFrame


# --- Taxonomy guidelines for "sample fields" ---
#
# We've categorized the fields on a prepped training sample into "usage classes":
#
# 1. Model inputs: fields consumed by the forward pass (WalkFrame: obs, board_mask).
# 2. Supervision: fields encoding a target or used by the loss function, in at least
#     one supported config (FrameSupervision).
# 3. Meta: everything else: provenance, metrics inputs (FrameMeta).
#
# The ordering of these classes reflects how fundamental the dependence is. Model
# inputs are required wherever the model is called (e.g. training, inference),
# supervision fields are only needed for training loss or measurement, and meta
# fields do not impact the resulting weights at all.
#
# Every sample field is assigned to a single usage class. A field is assigned to
# the most fundamental class that it qualifies for.
#
# Additional notes:
#
# - alive_mask is the first example of the multi-usage rule. The time_bin elim loss
#     uses it, placing it in supervision. But under some configs, it is metrics-only,
#     which would make it a meta field. As supervision is more fundamental, it wins,
#     and we designate `alive_mask` as a supervision field.
# - next_elim_dt (in aux_head_targets) should be a meta field, per these guidelines.
#     However, it is currently on FrameSupervision, as the aux-heads code needs a
#     bit of restructuring before we can fix this.
#     See a related TODO in `aux_heads/__init__.py`.
#
# Historical note and field inventory: 8.31-1-sample-fields-taxonomy.md.
# -----------------------------------------------


@dataclass(frozen=True)
class FrameMeta:
    frame_t: torch.Tensor
    players_alive: torch.Tensor
    p_start: torch.Tensor
    sample_idx: torch.Tensor

    def shallow_dict(self) -> dict[str, torch.Tensor]:
        return dict(vars(self))


# TODO(idea): Try out typing the batch dict. The training code passes the batch
# around as anonymous `dict[str, Tensor]`. A read-only TypedDict describing
# to_dict's output could replace those annotations with a named type and give us
# type-checked key access. It would be hand-maintained here, right next to
# TrainingSample, which should reduce the risk of drift. Since the batch is one
# flat dict, this would be a single flat analog of the whole sample.
# The main wrinkle: the aux heads' dynamically-keyed targets (see the TODO
# in aux_heads/__init__.py).
#
# A natural follow-on would be to use a similar strategy on the usage-class
# slices of a batched sample (model inputs, supervision, meta). Despite the
# batch being one flat dict, subset TypedDicts work as slice views directly on it.
# Pyright accepts the full dict where a subset view is expected (width subtyping).
#
# This todo comment emerges out of a related spike. Below, we document our TypedDict
# learnings from that spike:
#
# - Robustly checked, no casts: literal construction, subscript reads and writes
#   (unknown keys are errors), and iteration (`for k in d: d[k]` typechecks —
#   keys infer as the literal-key union).
# - `.get()` caveat: unknown keys are NOT flagged (returns Any | None). Subscript
#   required keys; save .get for NotRequired ones. closed=True (PEP 728,
#   experimental) does not close this hole.
# - Not assignable to dict[str, X] or Mapping[str, X] params. Annotating
#   consumers with the named type avoids casts entirely; casts remain only where
#   dynamically-keyed dicts merge in.
# - ReadOnly[...] items work (stdlib). Literal construction is allowed; for
#   incremental building, a builder subclass re-declaring just the conditional
#   keys as writable returns as the public type, cast-free.


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
    supervision: FrameSupervision,
    frame_meta: FrameMeta | None,
    sim_frame: SimFrame | None,
) -> TrainingSample:
    return TrainingSample(
        obs=torch.from_numpy(frame.obs),
        mask=torch.from_numpy(supervision.legality_mask),
        valid_mask=torch.from_numpy(frame.board_mask),
        action_target=torch.from_numpy(supervision.action_target),
        is_pass=torch.from_numpy(supervision.is_pass),
        value_target=torch.from_numpy(supervision.value_target),
        frame_meta=frame_meta,
        sim_frame=sim_frame,
        alive_mask=(
            None
            if supervision.alive_mask is None
            else torch.from_numpy(supervision.alive_mask)
        ),
        aux_head_targets=(
            None
            if supervision.aux_head_targets is None
            else {
                key: torch.from_numpy(value)
                for key, value in supervision.aux_head_targets.items()
            }
        ),
    )
