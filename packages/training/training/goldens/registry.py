from __future__ import annotations

from collections.abc import Hashable
from dataclasses import MISSING, dataclass, fields
from enum import Enum

from training.bc.aux_heads.elim_head_meta import ElimHeadVariant
from training.bc.config.metrics_config import MetricsConfig
from training.bc.config.targets_config import TargetsConfig
from training.bc.datapipe.emit_spec import PartialEmitSpec, partial_emit_spec_from
from training.bc.obs_config import ObsConfig
from training.goldens.store import REP_SEP, RefId, Surface


# The values of a key's dependency fields, taken from one point's config.
GroupId = tuple[Hashable, ...]


# --- Types ---


class RefForm(Enum):
    FULL = "full"                  # the array as emitted
    FRAME_HASHES = "frame_hashes"  # one hash per row along the first axis


@dataclass(frozen=True, kw_only=True)
class FixtureRecord:
    replay_id: str
    perspective_slot: int
    note: str

    @property
    def id(self) -> str:
        return f"{self.replay_id}-s{self.perspective_slot}"


@dataclass(frozen=True, kw_only=True)
class ObsPoint:
    name: str
    cfg: ObsConfig


@dataclass(frozen=True, kw_only=True)
class SupervisionPoint:
    name: str
    cfg: TargetsConfig
    keys: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class SupervisionKey:
    name: str
    # TargetsConfig fields that are inputs for the key (it "depends" on them).
    deps: tuple[str, ...]
    form: RefForm


# --- Registry data ---
#
# A point is a config value: the stored config for one guarded surface. Configs
# are written out as literals, never as a reference to a named constant such as
# OBS_CONFIG_DEFAULTS, so that a point stays fixed when prod's defaults move.

OBS_POINTS = [
    ObsPoint(
        name="fp16_player_status_on",
        cfg=ObsConfig(dense_history_n=5, obs_dtype="fp16", player_status_channels=True),
    ),
    ObsPoint(
        name="fp32_pre_player_status",
        cfg=ObsConfig(dense_history_n=5, obs_dtype="fp32", player_status_channels=False),
    ),
]

# The metrics surface is not guarded for now. The focus is obs and targets.
# We need a MetricsConfig to run the guarded code, so we use an "inert" one.
_INERT_METRICS_CFG = MetricsConfig(include_alive_mask=False)

# NOTE: This is intended to be append-only. Reference filenames depend on the order.
# A key group's files are named after its first point in this list. See
# `KeyGroup.representative` below.
# If the points were re-ordered, regen would report they had "moved" and we would
# need to re-bless.
SUPERVISION_POINTS = [
    SupervisionPoint(
        name="core",
        cfg=TargetsConfig(elim_variant=None, elim_bin_edges=None),
        keys=("legality_mask", "action_target", "is_pass", "value_target"),
    ),
    SupervisionPoint(
        name="time_bin",
        # TODO(sweep): confirm the edges against the checkpoints that trained
        # with the elim head (8.12-2 §9). Currently the ModelConfig default.
        cfg=TargetsConfig(
            elim_variant=ElimHeadVariant.TIME_BIN,
            elim_bin_edges=(10, 20, 40, 80, 160, 320, 640),
        ),
        keys=(
            "legality_mask", "action_target", "is_pass", "value_target",
            "alive_mask", "elim_bin_target",
        ),
    ),
    SupervisionPoint(
        name="next_death",
        cfg=TargetsConfig(elim_variant=ElimHeadVariant.NEXT_DEATH, elim_bin_edges=None),
        keys=(
            "legality_mask", "action_target", "is_pass", "value_target",
            "next_elim_target", "next_elim_dt", "next_elim_removal_dt", "present_mask",
        ),
    ),
]

SUPERVISION_KEYS = [
    SupervisionKey(name="legality_mask", deps=(), form=RefForm.FRAME_HASHES),
    SupervisionKey(name="action_target", deps=(), form=RefForm.FULL),
    SupervisionKey(name="is_pass", deps=(), form=RefForm.FULL),
    SupervisionKey(name="value_target", deps=(), form=RefForm.FULL),
    SupervisionKey(name="alive_mask", deps=(), form=RefForm.FULL),
    SupervisionKey(
        name="elim_bin_target",
        deps=("elim_variant", "elim_bin_edges"),
        form=RefForm.FULL,
    ),
    SupervisionKey(name="next_elim_target", deps=("elim_variant",), form=RefForm.FULL),
    SupervisionKey(name="next_elim_dt", deps=("elim_variant",), form=RefForm.FULL),
    SupervisionKey(name="next_elim_removal_dt", deps=("elim_variant",), form=RefForm.FULL),
    SupervisionKey(name="present_mask", deps=("elim_variant",), form=RefForm.FULL),
]

FIXTURES = [
    FixtureRecord(
        replay_id="ukRz7oSS8", perspective_slot=7, note="throwaway pick, eliminated at t=138"
    ),
    FixtureRecord(
        replay_id="xdZsRyX0O", perspective_slot=2, note="throwaway pick, eliminated at t=183"
    ),
    FixtureRecord(
        replay_id="NAT6qThbE", perspective_slot=2, note="throwaway pick, survivor"
    ),
]


# --- Entries ---


@dataclass(frozen=True)
class KeyRef:
    key: str
    form: RefForm
    ref: RefId


def partial_spec_for(point: SupervisionPoint) -> PartialEmitSpec:
    return partial_emit_spec_from(point.cfg, _INERT_METRICS_CFG)


@dataclass(frozen=True)
class ObsEntry:
    point: ObsPoint
    fixture: FixtureRecord
    ref: RefId

    @property
    def id(self) -> str:
        return f"{self.point.name}-{self.fixture.id}"


@dataclass(frozen=True)
class SupervisionEntry:
    point: SupervisionPoint
    fixture: FixtureRecord
    refs: dict[str, KeyRef]

    @property
    def id(self) -> str:
        return f"{self.point.name}-{self.fixture.id}"

    @property
    def spec(self) -> PartialEmitSpec:
        return partial_spec_for(self.point)


def group_id(key: SupervisionKey, cfg: TargetsConfig) -> GroupId:
    return tuple(getattr(cfg, f) for f in key.deps)


# The points that emit a key and agree on its deps. They share one reference
# file per fixture.
@dataclass(frozen=True)
class KeyGroup:
    key: SupervisionKey
    gid: GroupId
    points: tuple[str, ...]   # registry order

    # The group's first registered point. Its files are named after this point.
    @property
    def representative(self) -> str:
        return self.points[0]

    def ref(self, fixture: FixtureRecord) -> RefId:
        return RefId(
            Surface.SUPERVISION, point=self.representative, fixture=fixture.id, key=self.key.name
        )


def key_groups() -> list[KeyGroup]:
    out = []
    for key in SUPERVISION_KEYS:
        by_gid: dict[GroupId, list[str]] = {}
        for p in SUPERVISION_POINTS:
            if key.name in p.keys:
                by_gid.setdefault(group_id(key, p.cfg), []).append(p.name)
        out += [KeyGroup(key=key, gid=gid, points=tuple(points)) for gid, points in by_gid.items()]
    return out


def obs_ref(point: ObsPoint, fixture: FixtureRecord) -> RefId:
    return RefId(Surface.OBS, point=point.name, fixture=fixture.id)


def obs_entries() -> list[ObsEntry]:
    return [
        ObsEntry(point=p, fixture=fx, ref=obs_ref(p, fx))
        for p in OBS_POINTS
        for fx in FIXTURES
    ]


def supervision_entries() -> list[SupervisionEntry]:
    group_of = {(g.key.name, p): g for g in key_groups() for p in g.points}
    out = []
    for p in SUPERVISION_POINTS:
        for fx in FIXTURES:
            refs = {}
            for name in p.keys:
                g = group_of[(name, p.name)]
                refs[name] = KeyRef(key=name, form=g.key.form, ref=g.ref(fx))
            out.append(SupervisionEntry(point=p, fixture=fx, refs=refs))
    return out


# --- Import-time checks ---


def _assert_no_defaults(cls: type) -> None:
    for f in fields(cls):
        assert f.default is MISSING and f.default_factory is MISSING, (
            f"{cls.__name__}.{f.name} has a default; the registry relies on explicit construction"
        )


def _validate() -> None:
    for cls in (TargetsConfig, ObsConfig, MetricsConfig):
        _assert_no_defaults(cls)

    for names in (
        [p.name for p in OBS_POINTS],
        [p.name for p in SUPERVISION_POINTS],
        [k.name for k in SUPERVISION_KEYS],
        [fx.id for fx in FIXTURES],
    ):
        assert len(names) == len(set(names)), f"duplicate names: {names}"
        assert not any(REP_SEP in n for n in names), f"name contains {REP_SEP!r}: {names}"

    declared = set(_KEYS_BY_NAME)
    emitted = {k for p in SUPERVISION_POINTS for k in p.keys}
    assert emitted == declared, (
        f"keysets vs SUPERVISION_KEYS: missing {emitted - declared}, unused {declared - emitted}"
    )

    dep_fields = {f.name for f in fields(TargetsConfig)}
    for key in SUPERVISION_KEYS:
        bad = set(key.deps) - dep_fields
        assert not bad, f"{key.name}: unknown TargetsConfig fields {bad}"

    for p in SUPERVISION_POINTS:
        assert ("alive_mask" in p.keys) == partial_spec_for(p).emit_alive_mask, (
            f"{p.name}: alive_mask in keys must match the derived emit_alive_mask"
        )


_KEYS_BY_NAME = {k.name: k for k in SUPERVISION_KEYS}
_validate()
