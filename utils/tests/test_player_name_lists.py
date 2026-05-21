import logging

from utils.player_name_lists import load_players, load_players_raw, load_union


# --- load_players_raw / load_players ---

def test_raw_does_not_filter(tmp_path):
    p = tmp_path / "players.txt"
    p.write_text("alice\nbad\tname\nbob\n", encoding="utf-8")
    # Raw reader keeps invalid names intact for audit-style callers.
    assert load_players_raw(p) == ["alice", "bad\tname", "bob"]


def test_happy_path(tmp_path):
    p = tmp_path / "players.txt"
    p.write_text("alice\nbob\ncharlie\n", encoding="utf-8")
    assert load_players(p) == ["alice", "bob", "charlie"]


def test_strips_and_skips_blanks(tmp_path):
    p = tmp_path / "players.txt"
    p.write_text("  alice  \n\n   \nbob\n", encoding="utf-8")
    assert load_players(p) == ["alice", "bob"]


def test_filters_invalid_and_warns(tmp_path, caplog):
    p = tmp_path / "players.txt"
    # Embedded tab survives strip (only edges are stripped) and isn't split
    # by splitlines — filter_valid drops it with a warning.
    p.write_text("alice\nbad\tname\nbob\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="replay_collector.usernames"):
        result = load_players(p)
    assert result == ["alice", "bob"]
    assert len(caplog.records) == 1


def test_reads_utf8(tmp_path):
    p = tmp_path / "players.txt"
    nfc_name = "caf" + chr(0x00e9)  # precomposed é
    p.write_text(f"alice\n{nfc_name}\n", encoding="utf-8")
    assert load_players(p) == ["alice", nfc_name]


# --- load_union ---

def test_union_happy_path(tmp_path):
    (tmp_path / "list1.txt").write_text("alice\nbob\n", encoding="utf-8")
    (tmp_path / "list2.txt").write_text("charlie\ndave\n", encoding="utf-8")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("list1.txt\nlist2.txt\n", encoding="utf-8")
    assert load_union(manifest, tmp_path) == ["alice", "bob", "charlie", "dave"]


def test_union_dedup_first_seen(tmp_path):
    (tmp_path / "list1.txt").write_text("alice\nbob\n", encoding="utf-8")
    (tmp_path / "list2.txt").write_text("bob\ncharlie\n", encoding="utf-8")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("list1.txt\nlist2.txt\n", encoding="utf-8")
    # bob first appears in list1; list2's bob is dropped.
    assert load_union(manifest, tmp_path) == ["alice", "bob", "charlie"]


def test_union_missing_file_warns_and_skips(tmp_path, capsys):
    (tmp_path / "list1.txt").write_text("alice\n", encoding="utf-8")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("list1.txt\nmissing.txt\n", encoding="utf-8")
    result = load_union(manifest, tmp_path)
    assert result == ["alice"]
    captured = capsys.readouterr()
    assert "WARN missing" in captured.err
    assert "missing.txt" in captured.err
