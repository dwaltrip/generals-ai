import logging

import pytest

from replay_collector.usernames import display_name, filter_valid, is_valid_username


# ---------- is_valid_username ----------


class TestIsValidUsername:
    def test_normal_ascii(self):
        assert is_valid_username("alice")

    def test_internal_space(self):
        # generals.io usernames may contain internal spaces.
        assert is_valid_username("alice bob")

    def test_emoji_zwj_sequence(self):
        # The one ZWJ sequence in our DB: eye-in-speech-bubble. Legit.
        assert is_valid_username("\U0001f441‍\U0001f5e8")

    def test_empty_rejected(self):
        assert not is_valid_username("")

    @pytest.mark.parametrize(
        "ch",
        ["\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85",
         chr(0x2028), chr(0x2029), "\t"],
    )
    def test_layout_breaking_chars_rejected(self, ch):
        assert not is_valid_username(f"foo{ch}bar")

    @pytest.mark.parametrize(
        "ch",
        # Control chars (category Cc) beyond the layout-breaking set above.
        # NUL, BEL, BS, ESC, DEL, sample C1 control. These pass through
        # category=='Cc' rather than the explicit line/bidi sets.
        ["\x00", "\x07", "\x08", "\x1b", "\x7f", "\x90"],
    )
    def test_other_control_chars_rejected(self, ch):
        assert not is_valid_username(f"foo{ch}bar")

    @pytest.mark.parametrize(
        "ch",
        [chr(c) for c in (0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
                          0x2066, 0x2067, 0x2068, 0x2069)],
    )
    def test_bidi_format_chars_rejected(self, ch):
        assert not is_valid_username(f"foo{ch}bar")

    def test_leading_whitespace_rejected(self):
        assert not is_valid_username(" alice")

    def test_trailing_whitespace_rejected(self):
        assert not is_valid_username("alice ")

    def test_non_nfc_rejected(self):
        # NFD: "cafe" + U+0301 combining acute.
        assert not is_valid_username("cafe" + chr(0x0301))

    def test_nfc_accepted(self):
        # NFC: precomposed U+00E9 ("é").
        assert is_valid_username("caf" + chr(0x00e9))

    @pytest.mark.parametrize(
        "prefix",
        ["bc", "qg", "ad", "79", "ek", "20", "31", "ok", "at", "mx", "hi", "sa"],
    )
    def test_known_db_anomalies_rejected(self, prefix):
        # The 12 historical layout-breaking names in our DB share suffix \n\n\n\r\n.
        assert not is_valid_username(f"{prefix}\n\n\n\r\n")


# ---------- filter_valid ----------


class TestFilterValid:
    def test_passes_through_valid(self):
        assert filter_valid(["alice", "bob"]) == ["alice", "bob"]

    def test_drops_invalid(self):
        assert filter_valid(["alice", "bad\nname", "bob"]) == ["alice", "bob"]

    def test_empty_input(self):
        assert filter_valid([]) == []

    def test_warns_on_every_invalid(self, caplog):
        with caplog.at_level(logging.WARNING, logger="replay_collector.usernames"):
            filter_valid(["bad\nfoo", "bad\nbar", "bad\nfoo"])
        assert len(caplog.records) == 3

    def test_warning_has_no_literal_control_chars(self, caplog):
        # The whole point of %r — control chars must be escaped, not literal.
        with caplog.at_level(logging.WARNING, logger="replay_collector.usernames"):
            filter_valid(["bc\n\n\n\r\n"])
        msg = caplog.records[0].getMessage()
        assert "\n" not in msg
        assert "\r" not in msg
        assert "\\n" in msg


# ---------- display_name ----------


class TestDisplayName:
    def test_valid_returns_plain(self):
        assert display_name("alice") == "alice"

    def test_valid_with_internal_space(self):
        assert display_name("alice bob") == "alice bob"

    def test_invalid_returns_repr(self):
        assert display_name("bc\n\n\n\r\n") == "'bc\\n\\n\\n\\r\\n'"

    def test_empty_returns_repr(self):
        assert display_name("") == "''"
