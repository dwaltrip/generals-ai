import datetime as dt

import pytest

from replay_collector.cli.sweep_metadata import _parse_cutoff


def test_parse_cutoff_naive_uses_local_tz():
    expected = int(
        dt.datetime(2025, 1, 15, 14, 30, 0).astimezone().timestamp() * 1000
    )
    assert _parse_cutoff("2025-01-15T14:30:00") == expected


def test_parse_cutoff_with_offset():
    expected = int(
        dt.datetime(2025, 1, 15, 21, 30, 0, tzinfo=dt.timezone.utc).timestamp() * 1000
    )
    assert _parse_cutoff("2025-01-15T14:30:00-07:00") == expected


def test_parse_cutoff_z_suffix():
    expected = int(
        dt.datetime(2025, 1, 15, 21, 30, 0, tzinfo=dt.timezone.utc).timestamp() * 1000
    )
    assert _parse_cutoff("2025-01-15T21:30:00Z") == expected


def test_parse_cutoff_invalid_raises():
    with pytest.raises(ValueError):
        _parse_cutoff("not a date")
