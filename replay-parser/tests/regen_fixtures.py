"""Regenerate expected.npz for all fixtures.

Reads each fixture's wire.bin, runs `parse_replay`, derives the per-fixture
snapshot indices (standard + per-scenario), packs the oracle dict, and
writes expected.npz (compressed).

Usage (from replay-parser/):
    uv run python tests/regen_fixtures.py

A failing `test_sim_integration` means sim behavior changed. The discipline
rule: understand *why* fixtures fail before regenerating. Inspect the diff
between fresh sim output and the recorded `expected.npz` before running
this script.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _fixture_lib import FIXTURES, FIXTURES_DIR, fixture_snap_indices, pack_sim_output
from replay_parser.parser import parse_replay


def regen_one(spec) -> bool:
    fixture_dir = FIXTURES_DIR / spec.name
    wire_path = fixture_dir / "wire.bin"
    if not wire_path.exists():
        print(f"  SKIP {spec.name:<30} wire.bin missing — run seed_fixtures.py first")
        return False

    state, replay = parse_replay(wire_path.read_bytes())
    snap_indices = fixture_snap_indices(state, replay, spec.scenario)
    out = pack_sim_output(state, replay, snap_indices)

    npz_path = fixture_dir / "expected.npz"
    np.savez_compressed(npz_path, **out)
    size_kb = npz_path.stat().st_size / 1024
    print(
        f"  OK   {spec.name:<30} T={state.timestep:<5} "
        f"snaps={len(snap_indices):<3} size={size_kb:.1f}KB"
    )
    return True


def main() -> int:
    n_ok = sum(regen_one(f) for f in FIXTURES)
    print(f"\n{n_ok}/{len(FIXTURES)} fixtures regenerated")
    return 0 if n_ok == len(FIXTURES) else 1


if __name__ == "__main__":
    sys.exit(main())
