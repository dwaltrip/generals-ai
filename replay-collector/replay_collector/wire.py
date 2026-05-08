"""Encode/decode helpers for the `wire_data` BLOB column.

The wire-shape positional array (the value returned by
`generals_api.decompress_gior` on a freshly-downloaded `.gior`) is the
canonical form of a fetched replay. This module is the single place that
knows how it's serialized at rest, so collector save path, backfill, and
parser all share one path.

Storage format: gzip-compressed UTF-8 minified JSON. ~10 KB per replay
(roughly the same on-disk size as the legacy lz-string `raw` blob), and
~24x faster to decode than the `decompress_gior` round-trip. Stdlib only
— no third-party dependency.
"""

import gzip
import json


def encode(wire: list) -> bytes:
    """Serialize a wire-shape positional array to a gzip+JSON BLOB."""
    return gzip.compress(json.dumps(wire, separators=(",", ":")).encode("utf-8"))


def decode(blob: bytes) -> list:
    """Deserialize a wire-shape positional array from a gzip+JSON BLOB."""
    return json.loads(gzip.decompress(blob))
