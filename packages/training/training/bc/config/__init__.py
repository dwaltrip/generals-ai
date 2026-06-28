"""Versioned config resolution for checkpoints.

`resolve_config` is the migrate-once seam on the read path: a stored config
block of any past version is migrated to the current shape and constructed into a
`TrainConfig`, so no other code branches on a config version.

`stringify_paths` / `rewrap_paths` are the matched serialization pair for
`TrainConfig`'s `Path` fields. Checkpoints load with `weights_only=True`, which
rejects `Path` objects, so the paths are stored on disk as plain strings and
re-wrapped on read.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, get_type_hints

from training.bc.loss import LossConfig
from training.bc.model_config import build_model_cfg
from training.bc.train_config import TrainConfig


CONFIG_VERSION = 1

MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def migrate(config: dict[str, Any]) -> dict[str, Any]:
    """Step a stored config dict up to `CONFIG_VERSION`.

    Applies each registered migration in order — a no-op while none are
    registered. A config written by newer code (a version above `CONFIG_VERSION`)
    is rejected rather than silently mis-read. Returns a fresh dict; the input is
    untouched.
    """
    version = config["config_version"]
    if version > CONFIG_VERSION:
        raise ValueError(
            f"config_version {version} exceeds supported {CONFIG_VERSION} "
            "(checkpoint written by newer code)"
        )
    data = dict(config)
    for v in range(version, CONFIG_VERSION):
        data = MIGRATIONS[v](data)
    return data


def resolve_config(config: dict[str, Any]) -> TrainConfig:
    """Build a `TrainConfig` from a stored config block.

    `config` is a checkpoint's config block: an `asdict(TrainConfig)` plus
    `config_version`. Migrate the dict to the current version's shape, then
    construct the `TrainConfig` from it.
    """
    data = migrate(config)
    data.pop("config_version")
    arch = build_model_cfg(**data.pop("arch"))
    loss = LossConfig(**data.pop("loss"))
    rewrap_paths(data)
    return TrainConfig(arch=arch, loss=loss, **data)


# TrainConfig's Path-typed fields, found once at import so the serde pair picks
# up a new Path field without a hardcoded name list.
_PATH_FIELDS: tuple[str, ...] = tuple(
    name for name, hint in get_type_hints(TrainConfig).items() if hint is Path
)


def stringify_paths(data: dict[str, Any]) -> None:
    """Stringify `TrainConfig`'s `Path` fields present in `data`, in place."""
    for name in _PATH_FIELDS:
        if name in data:
            data[name] = str(data[name])


def rewrap_paths(data: dict[str, Any]) -> None:
    """Re-wrap `data`'s stringified `Path` fields back to `Path`, in place."""
    for name in _PATH_FIELDS:
        if name in data:
            data[name] = Path(data[name])
