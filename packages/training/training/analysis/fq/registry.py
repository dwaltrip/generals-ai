"""The `TableRegistry` type — a name -> FrameTableSpec catalog.

fq provides the type; the contents are domain. The families package builds an
instance (registering each family's specs) and hands it to `build_cli`, keeping
the library free of domain-specific specs.
"""

from __future__ import annotations

from training.analysis.fq.frame_table import FrameTableSpec


class TableRegistry:
    """A small name -> FrameTableSpec map an entry point hands to `build_cli`."""

    def __init__(self) -> None:
        self._specs: dict[str, FrameTableSpec] = {}

    def register(self, spec: FrameTableSpec) -> FrameTableSpec:
        """Add a spec under its `name`; returns it so callers can register
        inline. Rejects a duplicate name rather than silently shadowing."""
        if spec.name in self._specs:
            raise ValueError(f"duplicate table name {spec.name!r}")
        self._specs[spec.name] = spec
        return spec

    def __getitem__(self, name: str) -> FrameTableSpec:
        return self._specs[name]

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> list[str]:
        return sorted(self._specs)
