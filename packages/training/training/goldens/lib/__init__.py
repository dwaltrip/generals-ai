"""The goldens library's public API. Project code imports from here."""

from __future__ import annotations

from training.goldens.lib import store
from training.goldens.lib.apply import apply_regen
from training.goldens.lib.cli import regen_main
from training.goldens.lib.compute import compute_all
from training.goldens.lib.content import EntryContent, RowHashes
from training.goldens.lib.hashing import StreamingRowHasher, row_hashes
from training.goldens.lib.plan import RegenPlan, plan_regen
from training.goldens.lib.registry import Entry, Point, Registry, Surface, entries, fixture_infos
from training.goldens.lib.report import render_regen_report
from training.goldens.lib.run import run_regen
from training.goldens.lib.testing import (
    GoldenTestContext,
    assert_entry_matches,
    entry_test_id,
    load_test_context,
)


# TODO: As the number of exported names grows, we may want to group them under
# submodules, e.g. `testing.assert_entry_matches`. This may also apply to the
# other categories below.
__all__ = [
    # Registry
    "Entry",
    "Point",
    "Registry",
    "Surface",
    "entries",
    "fixture_infos",
    # Content and hashing
    "EntryContent",
    "RowHashes",
    "StreamingRowHasher",
    "row_hashes",
    # The references tree
    "store",
    # Regen
    "RegenPlan",
    "apply_regen",
    "compute_all",
    "plan_regen",
    "regen_main",
    "render_regen_report",
    "run_regen",
    # Testing
    "GoldenTestContext",
    "assert_entry_matches",
    "entry_test_id",
    "load_test_context",
]
