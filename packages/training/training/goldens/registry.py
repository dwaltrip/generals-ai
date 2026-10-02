from __future__ import annotations

from dataclasses import MISSING, fields

from training.bc.aux_heads.elim_head_meta import ElimHeadVariant
from training.bc.config.metrics_config import MetricsConfig
from training.bc.config.targets_config import TargetsConfig
from training.bc.obs_config import ObsConfig
from training.goldens.fixtures import FixtureData, FixtureRecord, load_fixture
from training.goldens.lib import Point, Registry, Surface
from training.goldens.producers import produce_obs, produce_supervision


# A point is a config value: the stored config for one guarded surface. Configs
# are written out as literals, never as a reference to a named constant such as
# OBS_CONFIG_DEFAULTS, so that a point stays fixed when prod's defaults move.

OBS_POINTS = (
    Point(
        name="fp16_player_status_on",
        cfg=ObsConfig(dense_history_n=5, obs_dtype="fp16", player_status_channels=True),
    ),
    Point(
        name="fp32_pre_player_status",
        cfg=ObsConfig(dense_history_n=5, obs_dtype="fp32", player_status_channels=False),
    ),
)

SUPERVISION_POINTS = (
    Point(name="core", cfg=TargetsConfig(elim_variant=None, elim_bin_edges=None)),
    Point(
        name="time_bin",
        # TODO(sweep): confirm the edges against the checkpoints that trained
        # with the elim head (8.12-2 §9). Currently the ModelConfig default.
        cfg=TargetsConfig(
            elim_variant=ElimHeadVariant.TIME_BIN,
            elim_bin_edges=(10, 20, 40, 80, 160, 320, 640),
        ),
    ),
    Point(
        name="next_death",
        cfg=TargetsConfig(elim_variant=ElimHeadVariant.NEXT_DEATH, elim_bin_edges=None),
    ),
)

FIXTURES = (
    FixtureRecord(
        replay_id="ukRz7oSS8", perspective_slot=7, note="throwaway pick, eliminated at t=138"
    ),
    FixtureRecord(
        replay_id="xdZsRyX0O", perspective_slot=2, note="throwaway pick, eliminated at t=183"
    ),
    FixtureRecord(
        replay_id="NAT6qThbE", perspective_slot=2, note="throwaway pick, survivor"
    ),
)

REGISTRY: Registry[FixtureRecord, FixtureData] = Registry(
    surfaces=(
        Surface(name="obs", points=OBS_POINTS, produce=produce_obs),
        Surface(name="supervision", points=SUPERVISION_POINTS, produce=produce_supervision),
    ),
    fixtures=FIXTURES,
    load_fixture=load_fixture,
)


# --- Import-time checks ---


def _assert_no_defaults(cls: type) -> None:
    for f in fields(cls):
        assert f.default is MISSING and f.default_factory is MISSING, (
            f"{cls.__name__}.{f.name} has a default; the registry relies on explicit construction"
        )


for _cls in (TargetsConfig, ObsConfig, MetricsConfig):
    _assert_no_defaults(_cls)
