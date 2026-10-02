"""
Regenerate the golden references for every registry entry. Every fixture is
computed and planned before anything is written.

Run from packages/training:
    uv run python -m training.goldens.regen_cli [--dry-run]
"""

# NOTE: The CLI's logic lives in the library (lib/cli.py). This module only
# supplies the project's registry, references root, and fixture description.
# If the library keeps growing, or is packaged separately, the library could
# own the entry point too, loading the registry from an import path
# (e.g. `training.goldens.registry:REGISTRY`).

from __future__ import annotations

import sys

from training.goldens.fixtures import FixtureData, FixtureRecord
from training.goldens.lib import regen_main
from training.goldens.paths import REFERENCES_DIR
from training.goldens.registry import REGISTRY


def _describe(fx: FixtureRecord, data: FixtureData) -> str:
    return f"T={data.game.T} end_t={data.persp.end_t}  ({fx.note})"


if __name__ == "__main__":
    sys.exit(regen_main(REGISTRY, REFERENCES_DIR, describe_fixture=_describe))
