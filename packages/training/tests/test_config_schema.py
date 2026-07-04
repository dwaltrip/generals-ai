"""Contracts on the stored config block.

Shape: a config-shape change must come with a `CONFIG_VERSION` bump, a new
frozen `config_schema/vN.json`, and a migration. Without the bump, loading an
old checkpoint would silently fill any new config field with its live default
value. A migration would fill it with the frozen historical value instead.

Types: every config field type must survive `torch.load(weights_only=True)`,
which rejects non-allowlisted classes. An unsafe field (e.g. `Path | None`)
would write fine and then fail every load of the checkpoint."""

from __future__ import annotations

from dataclasses import is_dataclass
import json
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from training.bc import config as bc_config
from training.bc.config import CONFIG_VERSION, MIGRATIONS, serialize_config
from training.bc.config.keyset import derive_keyset
from training.bc.train_config import TrainConfig


SCHEMA_DIR = Path(bc_config.__file__).parent / "config_schema"


def _frozen_keysets() -> dict[int, list[str]]:
    return {
        int(p.stem[1:]): json.loads(p.read_text()) for p in SCHEMA_DIR.glob("v*.json")
    }


def test_live_block_matches_frozen_keyset():
    config = TrainConfig(manifest=Path("m"), intermediate=Path("i"), run_dir=Path("r"))
    derived = derive_keyset(serialize_config(config))
    assert derived == _frozen_keysets()[CONFIG_VERSION], (
        "config block shape changed without a CONFIG_VERSION bump — bump it, "
        "freeze the new shape (uv run python -m training.bc.config.keyset "
        "> .../config_schema/vN.json), and register a migration"
    )


def test_config_version_is_newest_frozen():
    assert CONFIG_VERSION == max(_frozen_keysets())


def test_migrations_cover_all_stamped_versions():
    # v0 is unstamped — served by the legacy read path, not a migration.
    assert set(MIGRATIONS) == set(range(1, CONFIG_VERSION))


# --- field types must survive a weights_only load ---

_SAFE_LEAF_TYPES = {str, int, float, bool, type(None)}


def _check_hint(hint: Any, path: str, errors: list[str]) -> None:
    if hint in _SAFE_LEAF_TYPES:
        return
    if is_dataclass(hint):
        _check_dataclass(hint, f"{path}.", errors)
        return
    origin = get_origin(hint)
    if origin in (Union, UnionType):  # covers both Optional[X] and `X | Y`
        for arg in get_args(hint):
            _check_hint(arg, path, errors)
        return
    if origin in (tuple, list, dict):
        for arg in get_args(hint):
            if arg is not Ellipsis:
                _check_hint(arg, path, errors)
        return
    errors.append(
        f"{path}: {hint!r} is not recognized as weights_only-safe. If "
        "torch.load(weights_only=True) accepts it, add it to this check's "
        "allowlist; otherwise store the field as a plain type or extend the "
        "Path serde."
    )


def _check_dataclass(cls: type, prefix: str, errors: list[str]) -> None:
    for name, hint in get_type_hints(cls).items():
        if prefix == "" and name in bc_config._PATH_FIELDS:
            continue  # Path fields the serde stringifies
        _check_hint(hint, f"{prefix}{name}", errors)


# NOTE: this check decomposes type hints, so it will reject a new, legitimate
# construct (e.g. Literal) until its allowlist learns it. Watch for that
# friction as the configs evolve — re-evaluate the test if it becomes a
# recurring cost.
def test_config_field_types_are_weights_only_safe():
    errors: list[str] = []
    _check_dataclass(TrainConfig, "", errors)
    assert not errors, "\n".join(errors)
