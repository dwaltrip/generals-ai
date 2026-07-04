"""Derive the key-set of a stored config block. The frozen files in
`config_schema/` each record the key-set of one config version."""

from __future__ import annotations

from typing import Any


def derive_keyset(block: dict[str, Any]) -> list[str]:
    """The block's keys as sorted dotted paths, nested dicts included.

    e.g. `["arch", "arch.obs", "arch.obs.dense_history_n", ...]`.
    """
    paths: list[str] = []
    for key, value in block.items():
        paths.append(key)
        if isinstance(value, dict):
            paths.extend(f"{key}.{sub}" for sub in derive_keyset(value))
    return sorted(paths)


if __name__ == "__main__":
    # Print the current config block's key-set as JSON. A version bump saves
    # this output as the next frozen file:
    #   uv run python -m training.bc.config.keyset > .../config_schema/vN.json
    import json
    from pathlib import Path

    from training.bc.config import serialize_config
    from training.bc.train_config import TrainConfig

    config = TrainConfig(manifest=Path("m"), intermediate=Path("i"), run_dir=Path("r"))
    print(json.dumps(derive_keyset(serialize_config(config)), indent=2))
