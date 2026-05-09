"""Username validity gate.

A name is "invalid" if it would either corrupt our text/log formats or be
unrecoverable through human-mediated workflows (non-NFC). Invalid names
are stored faithfully in the DB; we refuse to act on them as primary
targets. The gate is wired at:

  - txt-load and JSON-load boundaries — `filter_valid` drops with a warning
    (see _shared.load_players, build_player_list.load_users_from_json).
  - tabular display sites — `display_name` re-renders via repr() so embedded
    control chars don't corrupt alignment.

Two kinds of invalid:

  - Layout / rendering hazards. Three groups of chars are rejected:

      - Line-boundary chars — exactly str.splitlines()'s split set:
        \\n, \\r, \\v, \\f, \\x1c-\\x1e, \\x85, U+2028, U+2029. These
        would corrupt our line-oriented txt formats at parse time —
        players.txt is one username per line, and logs and tabular
        stdout share the assumption.
      - Other control chars (Unicode category Cc): \\t, NUL, DEL, BEL,
        ESC, the rest of C0/C1. These corrupt tabular alignment,
        terminal output, and CSV exports.
      - Bidi-format controls (U+202A–E, U+2066–9). These reorder
        rendering of subsequent text in terminals and CSV viewers.

  - Non-NFC names can't survive human/clipboard/spreadsheet round-trips.
    We don't normalize on read — the goal is faithful match against
    generals.io's byte-exact storage. NFD-bearing names show up in
    [INVALID-USERNAME] warnings as anomalies worth investigating.

See docs/2026-05/5.08-1-username-handling-design.md for the full policy
and the empirical findings that motivated it.
"""

import logging
import unicodedata
from collections.abc import Iterable

log = logging.getLogger(__name__)

# Line/paragraph separators (categories Zl/Zp). Listed explicitly because
# they're the only Z* chars we reject — regular space (Zs) is fine, since
# generals.io usernames may contain internal spaces.
#
#   U+2028  LS   line separator
#   U+2029  PS   paragraph separator
#
# Assembled via chr() so the source file stays pure-ASCII — some text-
# processing pipelines normalize literal occurrences of these away.
_LINE_SEPARATORS = frozenset((chr(0x2028), chr(0x2029)))

# Bidi format controls. These are category Cf, but we don't blanket-reject
# Cf — it includes ZWJ (U+200D), which is legitimate in emoji sequences
# (we have one ZWJ-bearing name in the DB: eye-in-speech-bubble). So the
# bidi controls are listed explicitly:
#
#   U+202A  LRE   left-to-right embedding
#   U+202B  RLE   right-to-left embedding
#   U+202C  PDF   pop directional formatting
#   U+202D  LRO   left-to-right override
#   U+202E  RLO   right-to-left override  (the "Trojan Source" char)
#   U+2066  LRI   left-to-right isolate
#   U+2067  RLI   right-to-left isolate
#   U+2068  FSI   first strong isolate
#   U+2069  PDI   pop directional isolate
#
# These can reorder rendering of *subsequent* text in a terminal or CSV
# viewer — the kind of bug you'd never trace back to a username.
_BIDI_FORMAT_CHARS = frozenset(
    chr(c) for c in (
        0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
        0x2066, 0x2067, 0x2068, 0x2069,
    )
)


def is_valid_username(name: str) -> bool:
    """True iff `name` is safe to use as a primary action target — non-empty,
    no edge whitespace, no control chars, no line/bidi format chars, and
    in NFC normal form."""
    if not name:
        return False
    if name != name.strip():
        return False
    for c in name:
        if unicodedata.category(c) == "Cc":
            return False
        if c in _LINE_SEPARATORS:
            return False
        if c in _BIDI_FORMAT_CHARS:
            return False
    if unicodedata.normalize("NFC", name) != name:
        return False
    return True


def filter_valid(names: Iterable[str]) -> list[str]:
    """Drop invalid names; warn on each one."""
    out = []
    for name in names:
        if is_valid_username(name):
            out.append(name)
        else:
            log.warning("[INVALID-USERNAME] skipping: %r", name)
    return out


def display_name(name: str) -> str:
    """Safe rendering for tabular output. Returns repr() for invalid names
    so embedded control chars don't corrupt the surrounding line."""
    return name if is_valid_username(name) else repr(name)
