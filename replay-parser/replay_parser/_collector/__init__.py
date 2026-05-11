"""Bridge to `replay_collector`.

All cross-package imports from `replay_collector` route through this
subpackage — one file per upstream module, re-exports only. The folder
serves as a living inventory of what `replay-parser` actually depends
on from its sibling, and centralizes the surface so a future extraction
to a shared library is mechanical.

Add a new file here only when the parser actually needs something from
the corresponding upstream module. Don't pre-mirror speculatively.
"""
