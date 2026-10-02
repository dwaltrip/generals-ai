from __future__ import annotations

import numpy as np

from training.bc.config.metrics_config import MetricsConfig
from training.bc.config.targets_config import TargetsConfig
from training.bc.datapipe.emit import emit_tail
from training.bc.datapipe.emit_spec import partial_emit_spec_from
from training.bc.datapipe.precompute import precompute_for
from training.bc.datapipe.walk import walk
from training.bc.obs_config import ObsConfig
from training.goldens.fixtures import FixtureData
from training.goldens.lib import EntryContent, StreamingRowHasher, row_hashes


# The metrics surface is not guarded for now. The focus is obs and targets.
# We need a MetricsConfig to run the guarded code, so we use an "inert" one.
_INERT_METRICS_CFG = MetricsConfig(include_alive_mask=False)


# The obs goldens guard the bytes of the obs tensor as produced during training.
# Hashes per frame and per channel are enough to robustly detect changes and tell
# us where to look (which ticks and channels were affected).
# Higher fidelity alternatives were considered (storing the full tensor or hashing
# frames x channels), but the storage cost would be burdensome for a small gain
# (e.g. knowing the exact frame on which a channel changed).
def produce_obs(cfg: ObsConfig, data: FixtureData) -> EntryContent:
    hasher = StreamingRowHasher(axes=(0, 1))
    for frame in walk(data.game, data.persp, cfg):
        hasher.add(frame.obs)
    frame_hashes, channel_hashes = hasher.hashes()
    return {"frame_hashes": frame_hashes, "channel_hashes": channel_hashes}


def produce_supervision(cfg: TargetsConfig, data: FixtureData) -> EntryContent:
    game, persp = data.game, data.persp
    assert persp.end_t > 0, f"perspective has zero frames (end_t={persp.end_t})"

    spec = partial_emit_spec_from(cfg, _INERT_METRICS_CFG)
    pre = precompute_for(spec, game)
    per_t = [emit_tail(game, t, persp, spec, pre).to_dict() for t in range(persp.end_t)]
    # Every key is stacked per frame over t, including scalars like value_target.
    stacked = {key: np.stack([d[key] for d in per_t]) for key in per_t[0]}

    content: EntryContent = dict(stacked)
    content["legality_mask"] = row_hashes(stacked["legality_mask"])
    return content
