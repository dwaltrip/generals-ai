import datetime as dt

import pytest

from replay_collector.cli.sweep_metadata import _iso_to_epoch_ms


def test_iso_to_epoch_ms_naive_uses_local_tz():
    expected = int(
        dt.datetime(2025, 1, 15, 14, 30, 0).astimezone().timestamp() * 1000
    )
    assert _iso_to_epoch_ms("2025-01-15T14:30:00") == expected


def test_iso_to_epoch_ms_with_offset():
    expected = int(
        dt.datetime(2025, 1, 15, 21, 30, 0, tzinfo=dt.UTC).timestamp() * 1000
    )
    assert _iso_to_epoch_ms("2025-01-15T14:30:00-07:00") == expected


def test_iso_to_epoch_ms_z_suffix():
    expected = int(
        dt.datetime(2025, 1, 15, 21, 30, 0, tzinfo=dt.UTC).timestamp() * 1000
    )
    assert _iso_to_epoch_ms("2025-01-15T21:30:00Z") == expected


def test_iso_to_epoch_ms_invalid_raises():
    with pytest.raises(ValueError):
        _iso_to_epoch_ms("not a date")
