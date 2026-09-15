from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any

from training.bc.aux_heads.elim_head_meta import ElimHeadVariant
from training.bc.config.metrics_config import MetricsConfig
from training.bc.config.targets_config import TargetsConfig
from training.bc.datapipe.emit_spec import PartialEmitSpec, partial_emit_spec_from
from training.bc.obs_config import ObsConfig


REP_SEP = "@"

DATA_ROOT = Path(__file__).resolve().parents[2] / "tests" / "goldens"
FIXTURES_DIR = DATA_ROOT / "fixtures"
REFERENCES_DIR = DATA_ROOT / "references"


# --- Types ---


class RefForm(Enum):
    FULL = "full"                  # the array as emitted
    FRAME_HASHES = "frame_hashes"  # one hash per row along the first axis


@dataclass(frozen=True, kw_only=True)
class FixtureRecord:
    replay_id: str
    slot: int
    note: str

    @property
    def id(self) -> str:
        return f"{self.replay_id}-s{self.slot}"


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
    # TargetsConfig fields this key's bytes depend on. Points that agree on
    # these fields share one stored reference. Regen verifies the claim.
    deps: tuple[str, ...]
    form: RefForm


# --- Registry data ---
#
# A point is a config value: the stored config for one guarded surface. Every
# config here is written out as a literal. Never reference a named config
# constant (e.g. OBS_CONFIG_DEFAULTS): the registry pins values so that prod's
# defaults can move without re-describing what is guarded.

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

# The metrics surface is not guarded. A point holds every config field that
# affects the bytes of an emitted key. Fields that only add or remove columns
# (metrics column requests) are held at their null value.
_METRICS = MetricsConfig(include_alive_mask=False)

# NOTE: This list is append only! Reference filenames depend on the order.
# The "representative point" for a key is the first one in this list that emits it.
# See `representative` below.
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
    SupervisionKey(name="elim_bin_target", deps=("elim_variant", "elim_bin_edges"), form=RefForm.FULL),
    SupervisionKey(name="next_elim_target", deps=("elim_variant",), form=RefForm.FULL),
    SupervisionKey(name="next_elim_dt", deps=("elim_variant",), form=RefForm.FULL),
    SupervisionKey(name="next_elim_removal_dt", deps=("elim_variant",), form=RefForm.FULL),
    SupervisionKey(name="present_mask", deps=("elim_variant",), form=RefForm.FULL),
]

FIXTURES = [
    FixtureRecord(replay_id="ukRz7oSS8", slot=7, note="throwaway pick, eliminated at t=138"),
    FixtureRecord(replay_id="xdZsRyX0O", slot=2, note="throwaway pick, eliminated at t=183"),
    FixtureRecord(replay_id="NAT6qThbE", slot=2, note="throwaway pick, survivor"),
]


# --- Entries ---


@dataclass(frozen=True)
class KeyRef:
    key: str
    form: RefForm
    path: Path


@dataclass(frozen=True)
class ObsEntry:
    point: str
    cfg: ObsConfig
    fixture: FixtureRecord
    ref_path: Path


@dataclass(frozen=True)
class SupervisionEntry:
    point: str
    cfg: TargetsConfig
    spec: PartialEmitSpec
    fixture: FixtureRecord
    refs: dict[str, KeyRef]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.refs)


def spec_for(point: SupervisionPoint) -> PartialEmitSpec:
    return partial_emit_spec_from(point.cfg, _METRICS)


def group_id(key: SupervisionKey, cfg: TargetsConfig) -> tuple[Any, ...]:
    return tuple(getattr(cfg, f) for f in key.deps)


def representative(key: SupervisionKey, gid: tuple[Any, ...]) -> str:
    for point in SUPERVISION_POINTS:
        if key.name in point.keys and group_id(key, point.cfg) == gid:
            return point.name
    raise KeyError(f"no point emits {key.name!r} with group {gid!r}")


def supervision_key(name: str) -> SupervisionKey:
    return _KEYS_BY_NAME[name]


def obs_ref_path(point: str, fixture: FixtureRecord) -> Path:
    return REFERENCES_DIR / "obs" / point / f"{fixture.id}.npz"


def supervision_ref_path(key: SupervisionKey, cfg: TargetsConfig, fixture: FixtureRecord) -> Path:
    rep = representative(key, group_id(key, cfg))
    return REFERENCES_DIR / "supervision" / fixture.id / f"{rep}{REP_SEP}{key.name}.npy"


def obs_entries() -> list[ObsEntry]:
    return [
        ObsEntry(point=p.name, cfg=p.cfg, fixture=fx, ref_path=obs_ref_path(p.name, fx))
        for p in OBS_POINTS
        for fx in FIXTURES
    ]


def supervision_entries() -> list[SupervisionEntry]:
    out = []
    for p in SUPERVISION_POINTS:
        spec = spec_for(p)
        for fx in FIXTURES:
            refs = {
                name: KeyRef(
                    key=name,
                    form=supervision_key(name).form,
                    path=supervision_ref_path(supervision_key(name), p.cfg, fx),
                )
                for name in p.keys
            }
            out.append(SupervisionEntry(point=p.name, cfg=p.cfg, spec=spec, fixture=fx, refs=refs))
    return out


# --- Import-time checks ---


def _assert_no_defaults(cls: type) -> None:
    for f in fields(cls):
        assert f.default is MISSING and f.default_factory is MISSING, (
            f"{cls.__name__}.{f.name} has a default; the registry relies on explicit construction"
        )


def _validate() -> None:
    for cls in (TargetsConfig, ObsConfig, MetricsConfig, PartialEmitSpec):
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
        assert ("alive_mask" in p.keys) == spec_for(p).emit_alive_mask, (
            f"{p.name}: alive_mask in keys must match the derived emit_alive_mask"
        )


_KEYS_BY_NAME = {k.name: k for k in SUPERVISION_KEYS}
_validate()
