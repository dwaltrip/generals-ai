from replay_collector import db, generals_api, sweep


def _listing(rid: str, started: int) -> dict:
    return {
        "id": rid,
        "started": started,
        "type": "1v1",
        "ladder_id": "duel",
        "turns": 100,
        "player_count": 2,
    }


def test_sweep_one_stops_on_recency_cutoff(monkeypatch):
    listings = [
        _listing("a", 1_000_000_000_000),
        _listing("b",   900_000_000_000),
        _listing("c",   500_000_000_000),  # first past cutoff — should NOT be walked
        _listing("d",   400_000_000_000),
    ]
    walked_ids: list[str] = []

    monkeypatch.setattr(generals_api, "user_exists", lambda client, u: True)
    monkeypatch.setattr(
        generals_api, "iter_user_replay_pages",
        lambda client, u: iter([listings]),
    )
    monkeypatch.setattr(
        db, "try_insert_listing",
        lambda entry: (walked_ids.append(entry["id"]) or True),
    )
    monkeypatch.setattr(db, "has_full_data", lambda rid: False)

    stats = sweep.sweep_one(
        client=None, username="alice", max_listings=1000,
        recency_cutoff_ms=600_000_000_000,
    )

    assert stats.stop_reason == "recency_cutoff"
    assert walked_ids == ["a", "b"]
    assert stats.listings_walked == 2


def test_sweep_one_no_cutoff_walks_all(monkeypatch):
    listings = [_listing("a", 100), _listing("b", 50)]

    monkeypatch.setattr(generals_api, "user_exists", lambda client, u: True)
    monkeypatch.setattr(
        generals_api, "iter_user_replay_pages",
        lambda client, u: iter([listings]),
    )
    monkeypatch.setattr(db, "try_insert_listing", lambda entry: True)
    monkeypatch.setattr(db, "has_full_data", lambda rid: False)

    stats = sweep.sweep_one(client=None, username="alice", max_listings=1000)

    assert stats.stop_reason == "exhausted"
    assert stats.listings_walked == 2
