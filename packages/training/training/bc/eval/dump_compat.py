"""Backward-compat readers for dump artifacts.

New dumps write canonical column names (the writer is `dump.py`); these helpers
let offline analysis read older dumps too, warning when a legacy name is hit so
stale-dump usage stays visible and countable. Torch-free on purpose, so pure-numpy
analysis modules can use it without pulling in the training stack.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np


log = logging.getLogger(__name__)


class _DumpLike(Protocol):
    """An `np.load` NpzFile or any dict-like keyed by column name."""

    def __contains__(self, key: str, /) -> bool: ...
    def __getitem__(self, key: str, /) -> np.ndarray: ...


# Pre-rename names for the per-player alive mask (the next_death / time_bin elim
# variants each had their own). New dumps write `alive_mask`; these are kept only
# to read historical dumps and can go once those have aged out.
_LEGACY_ALIVE_MASK_KEYS = ("next_elim_alive_mask", "elim_alive_mask")


def read_alive_mask(dump: _DumpLike) -> np.ndarray:
    """Read the per-player alive mask from a dump. Prefers the canonical
    `alive_mask`; for pre-rename dumps falls back to a legacy key and logs a
    WARNING per read (so how often old dumps are still in play is observable).
    Raises KeyError if no alive-mask column is present.
    """
    if "alive_mask" in dump:
        return dump["alive_mask"]
    for key in _LEGACY_ALIVE_MASK_KEYS:
        if key in dump:
            log.warning("dump: using legacy alive-mask key %r (canonical is 'alive_mask')", key)
            return dump[key]
    raise KeyError("dump has no alive-mask column (alive_mask or a legacy name)")
