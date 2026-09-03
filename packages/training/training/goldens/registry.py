from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any

from training.bc.config.targets_config import TargetsConfig
from training.bc.datapipe.emit_spec import PartialEmitSpec
from training.bc.obs_config import ObsConfig


REP_SEP = "@"

DATA_ROOT = Path(__file__).resolve().parents[2] / "tests" / "goldens"
FIXTURES_DIR = DATA_ROOT / "fixtures"
REFERENCES_DIR = DATA_ROOT / "references"


# --- Records ---


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
    cfg: dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class SupervisionPoint:
    name: str
    targets: dict[str, Any]
    emit_alive_mask: bool
    keys: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class SupervisionKey:
    name: str
    # Config fields this key's bytes depend on. Points that agree on these
    # fields share one stored reference. Regen verifies the claim.
    deps: tuple[str, ...]
    hashed: bool = False


# --- Data (plain values only) ---

OBS_POINTS = [
    ObsPoint(
        name="fp16_player_status_on",
        cfg={"dense_history_n": 5, "obs_dtype": "fp16", "player_status_channels": True},
    ),
    ObsPoint(
        name="fp32_pre_player_status",
        cfg={"dense_history_n": 5, "obs_dtype": "fp32", "player_status_channels": False},
    ),
]

# Append only: a group's reference file is named after the first point (in
# this order) that emits the key, so inserting a point above others relocates files.
SUPERVISION_POINTS = [
    SupervisionPoint(
        name="core",
        targets={"elim_variant": None, "elim_bin_edges": None},
        emit_alive_mask=False,
        keys=("legality_mask", "action_target", "is_pass", "value_target"),
    ),
    SupervisionPoint(
        name="alive_only",
        targets={"elim_variant": None, "elim_bin_edges": None},
        emit_alive_mask=True,
        keys=("legality_mask", "action_target", "is_pass", "value_target", "alive_mask"),
    ),
    SupervisionPoint(
        name="time_bin",
        # TODO(sweep): confirm the edges against the checkpoints that trained
        # with the elim head (8.12-2 §9). Currently the ModelConfig default.
        targets={"elim_variant": "time_bin", "elim_bin_edges": [10, 20, 40, 80, 160, 320, 640]},
        emit_alive_mask=True,
        keys=(
            "legality_mask", "action_target", "is_pass", "value_target",
            "alive_mask", "elim_bin_target",
        ),
    ),
    SupervisionPoint(
        name="next_death",
        targets={"elim_variant": "next_death", "elim_bin_edges": None},
        emit_alive_mask=True,
        keys=(
            "legality_mask", "action_target", "is_pass", "value_target",
            "alive_mask", "next_elim_target", "next_elim_dt",
            "next_elim_removal_dt", "present_mask",
        ),
    ),
]

SUPERVISION_KEYS = [
    SupervisionKey(name="legality_mask", deps=(), hashed=True),
    SupervisionKey(name="action_target", deps=()),
    SupervisionKey(name="is_pass", deps=()),
    SupervisionKey(name="value_target", deps=()),
    SupervisionKey(name="alive_mask", deps=()),
    SupervisionKey(name="elim_bin_target", deps=("elim_variant", "elim_bin_edges")),
    SupervisionKey(name="next_elim_target", deps=("elim_variant",)),
    SupervisionKey(name="next_elim_dt", deps=("elim_variant",)),
    SupervisionKey(name="next_elim_removal_dt", deps=("elim_variant",)),
    SupervisionKey(name="present_mask", deps=("elim_variant",)),
]

FIXTURES = [
    FixtureRecord(replay_id="ukRz7oSS8", slot=7, note="throwaway pick, eliminated at t=138"),
    FixtureRecord(replay_id="xdZsRyX0O", slot=2, note="throwaway pick, eliminated at t=183"),
    FixtureRecord(replay_id="NAT6qThbE", slot=2, note="throwaway pick, survivor"),
]


# --- Entries ---


@dataclass(frozen=True)
class ObsEntry:
    point: str
    cfg: ObsConfig
    fixture: FixtureRecord
    ref_path: Path


@dataclass(frozen=True)
class SupervisionEntry:
    point: str
    spec: PartialEmitSpec
    keys: tuple[str, ...]
    hashed_keys: frozenset[str]
    fixture: FixtureRecord
    paths: dict[str, Path]


def spec_for(point: SupervisionPoint) -> PartialEmitSpec:
    return PartialEmitSpec(
        targets=TargetsConfig(**point.targets),
        emit_alive_mask=point.emit_alive_mask,
        attach_sim_frame=False,
    )


def group_id(key: SupervisionKey, spec: PartialEmitSpec) -> tuple[Any, ...]:
    targets_fields = {f.name for f in fields(TargetsConfig)}
    return tuple(
        getattr(spec.targets, f) if f in targets_fields else getattr(spec, f)
        for f in key.deps
    )


def representative(key: SupervisionKey, gid: tuple[Any, ...]) -> str:
    for point in SUPERVISION_POINTS:
        if key.name in point.keys and group_id(key, spec_for(point)) == gid:
            return point.name
    raise KeyError(f"no point emits {key.name!r} with group {gid!r}")


def supervision_key(name: str) -> SupervisionKey:
    return _KEYS_BY_NAME[name]


def obs_ref_path(point: str, fixture: FixtureRecord) -> Path:
    return REFERENCES_DIR / "obs" / point / f"{fixture.id}.npz"


def supervision_ref_path(key: SupervisionKey, spec: PartialEmitSpec, fixture: FixtureRecord) -> Path:
    rep = representative(key, group_id(key, spec))
    return REFERENCES_DIR / "supervision" / fixture.id / f"{rep}{REP_SEP}{key.name}.npy"


def obs_entries() -> list[ObsEntry]:
    return [
        ObsEntry(
            point=p.name,
            cfg=ObsConfig(**p.cfg),
            fixture=fx,
            ref_path=obs_ref_path(p.name, fx),
        )
        for p in OBS_POINTS
        for fx in FIXTURES
    ]


def supervision_entries() -> list[SupervisionEntry]:
    out = []
    for p in SUPERVISION_POINTS:
        spec = spec_for(p)
        for fx in FIXTURES:
            out.append(
                SupervisionEntry(
                    point=p.name,
                    spec=spec,
                    keys=p.keys,
                    hashed_keys=frozenset(k for k in p.keys if supervision_key(k).hashed),
                    fixture=fx,
                    paths={k: supervision_ref_path(supervision_key(k), spec, fx) for k in p.keys},
                )
            )
    return out


# --- Import-time checks (pure, no IO) ---


def _assert_no_defaults(cls: type) -> None:
    for f in fields(cls):
        assert f.default is MISSING and f.default_factory is MISSING, (
            f"{cls.__name__}.{f.name} has a default; the registry relies on explicit construction"
        )


def _validate() -> None:
    for cls in (TargetsConfig, ObsConfig, PartialEmitSpec):
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

    dep_fields = {f.name for f in fields(TargetsConfig)} | {f.name for f in fields(PartialEmitSpec)}
    for key in SUPERVISION_KEYS:
        bad = set(key.deps) - dep_fields
        assert not bad, f"{key.name}: unknown dependency fields {bad}"

    for p in OBS_POINTS:
        ObsConfig(**p.cfg)
    for p in SUPERVISION_POINTS:
        spec_for(p)


_KEYS_BY_NAME = {k.name: k for k in SUPERVISION_KEYS}
_validate()
