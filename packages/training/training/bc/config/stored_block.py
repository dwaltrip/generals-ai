"""The checkpoint's stored config block: shape, versioning, and serde.

This module is the sole interface for writing and reading the raw stored config block
from a model checkpoint dict. By design, it is entirely agnostic and decoupled from
the rest of the checkpoint file and any other metadata contained within.
The interface: `TrainConfig <-> plain-valued config dict (StoredConfigBlock)`.
"""

from __future__ import annotations

from collections.abc import Callable
import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any, get_type_hints

from training.bc.loss import LossConfig
from training.bc.model_config import build_model_cfg
from training.bc.train_config import TrainConfig


CONFIG_VERSION = 1

# The "stored config block" for model checkpoints. It is a plain-valued dict
# that is serialized and stored directly in the checkpoint dict.
# If usage of this expands and we want better typing (e.g. live migrations, tooling
# that passes blocks around), we can upgrade to a wrapper dataclass holding
# `config_version` and a `data` dict.
# `TypedDict` (PEP 728) was tried and rejected: the copy / spread-heavy usages
# would cause awkward type casts to proliferate.
StoredConfigBlock = dict[str, Any]

MIGRATIONS: dict[int, Callable[[StoredConfigBlock], StoredConfigBlock]] = {}


def migrate(config: StoredConfigBlock) -> StoredConfigBlock:
    """Step a stored config block up to `CONFIG_VERSION` via the registered migrations.

    A version above `CONFIG_VERSION` raises `ValueError`: the config was written
    by newer code or the block was mis-edited. Either way, it can't be trusted.

    Returns a new dict, leaving the input untouched.
    """
    version = config["config_version"]
    if version > CONFIG_VERSION:
        raise ValueError(
            f"config_version {version} exceeds supported {CONFIG_VERSION} "
            "(checkpoint with an unknown version)"
        )
    # Deep copy so migrations may mutate nested structures freely.
    data = copy.deepcopy(config)
    for v in range(version, CONFIG_VERSION):
        data = MIGRATIONS[v](data)
    return data


def resolve_config(config: StoredConfigBlock) -> TrainConfig:
    """Build a `TrainConfig` from a stored config block of any supported version.

    The migrate-once seam on the read path: older blocks are migrated to a shape
    the current code supports, and downstream code can ignore that complexity.
    """
    # NOTE(ckpt-cfg-refactor-note): Currently, the only shape is `TrainConfig`.
    # Later, there may be dedicated back-compat shapes — enabling older checkpoints
    # to keep loading while "legacy" model and training code moves out of the main paths.
    data = migrate(config)
    data.pop("config_version")
    arch = build_model_cfg(**data.pop("arch"))
    loss = LossConfig(**data.pop("loss"))
    rewrap_paths(data)
    return TrainConfig(arch=arch, loss=loss, **data)


def serialize_config(config: TrainConfig) -> StoredConfigBlock:
    """Build a stored config block from a `TrainConfig` — the write-side
    counterpart of `resolve_config`."""
    block: StoredConfigBlock = {**asdict(config), "config_version": CONFIG_VERSION}
    stringify_paths(block)
    return block


# Checkpoints load with `weights_only=True`, which rejects `Path` objects, so
# `Path` fields are stored on disk as plain strings and re-wrapped on read.
#
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
