"""Deep-merge and path-expansion utilities for nested dicts.

Used by sweep tooling to compose config overlays from dotted-path axes
(e.g. ``arch.obs.dense_history_n``) with compound dict values.
"""

from __future__ import annotations

import copy
from typing import Any


def unflatten(path: str, value: Any) -> dict:
    """Expand a dotted path and value into a nested dict.

    >>> unflatten("a.b.c", 5)
    {'a': {'b': {'c': 5}}}
    """
    keys = path.split(".")
    result: dict = {}
    target = result
    for key in keys[:-1]:
        target[key] = {}
        target = target[key]
    target[keys[-1]] = value
    return result


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* into *base*, returning a new dict.

    - When both sides have a dict at the same key, merge recursively.
    - Otherwise the overlay value replaces the base value.
    - Neither input is mutated.
    """
    result = copy.copy(base)
    for key, overlay_val in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(overlay_val, dict):
            result[key] = deep_merge(result[key], overlay_val)
        else:
            result[key] = copy.deepcopy(overlay_val)
    return result
