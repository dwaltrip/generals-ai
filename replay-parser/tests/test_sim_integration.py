"""Sim integration tests against pre-recorded fixtures.

Each fixture pairs `wire.bin` (input) with `expected.npz` (oracle: snapshots
+ end-state + events + damage matrix). A test failure means sim output for
that replay diverged from the recorded oracle.

Regenerate fixtures with `tests/regen_fixtures.py` — but only after
investigating the failure. The fixtures exist to catch unintended
behavior changes; regenerating without diagnosis defeats that.
"""

import numpy as np
import pytest

from replay_parser.parser import parse_replay

from _fixture_lib import FIXTURES, FIXTURES_DIR, pack_sim_output


@pytest.mark.parametrize(
    "spec",
    FIXTURES,
    ids=[f.name for f in FIXTURES],
)
def test_fixture(spec):
    wire_path = FIXTURES_DIR / spec.name / "wire.bin"
    npz_path = FIXTURES_DIR / spec.name / "expected.npz"
    if not wire_path.exists():
        pytest.skip(f"missing wire.bin: {wire_path}")
    if not npz_path.exists():
        pytest.skip(f"missing expected.npz: {npz_path}")

    state, replay = parse_replay(wire_path.read_bytes())
    expected = dict(np.load(npz_path))
    snap_indices = expected["snap_indices"].tolist()
    actual = pack_sim_output(state, replay, snap_indices)

    extra = set(actual) - set(expected)
    missing = set(expected) - set(actual)
    assert not extra and not missing, f"key drift: extra={extra} missing={missing}"

    for key in expected:
        np.testing.assert_array_equal(
            actual[key], expected[key], err_msg=f"mismatch on key: {key}"
        )
