import json
import urllib.request

import lzstring

S3_BASE_URL = "https://generalsio-replays-na.s3.amazonaws.com"

_lzs = lzstring.LZString()


def decompress_gior(raw: bytes) -> list:
    # The .gior is what JS produces via LZString.compressFromUint8Array: each
    # 16-bit char is split into two big-endian bytes. Pair them back up and
    # hand the resulting string to plain decompress().
    s = "".join(chr((raw[i * 2] << 8) | raw[i * 2 + 1]) for i in range(len(raw) // 2))
    return json.loads(_lzs.decompress(s))


def fetch_replay(replay_id: str) -> list:
    url = f"{S3_BASE_URL}/{replay_id}.gior"
    with urllib.request.urlopen(url) as resp:
        raw = resp.read()
    return decompress_gior(raw)
