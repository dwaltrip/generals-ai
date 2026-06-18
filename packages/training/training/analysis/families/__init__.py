"""Analysis families: the domain catalog fq's generic CLI is handed.

Each family module (e.g. `elim_head_debug`) defines one or more `FrameTableSpec`s
plus their helpers; this package registers them into `REGISTRY` and names the
default split. `scripts/fq.py` hands both to `fq.cli.build_cli`.
"""

from __future__ import annotations

from settings import TRAINING_DATA_DIR
from training.analysis.argmin_toy.spec import ARGMIN_TOY
from training.analysis.families.elim_head_debug import ELIM_HEAD_DEBUG
from training.analysis.fq.registry import TableRegistry


DEFAULT_MANIFEST = TRAINING_DATA_DIR / "splits" / "cloud_24k_sub_11k.json"

REGISTRY = TableRegistry()
REGISTRY.register(ELIM_HEAD_DEBUG)
REGISTRY.register(ARGMIN_TOY)
