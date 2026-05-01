import itertools
import json
from typing import Iterator

import lzstring

from replay_collector.client import TrackedClient

API_BASE = "https://generals.io"
S3_BASE = "https://generalsio-replays-na.s3.amazonaws.com"
PAGE_SIZE = 200  # server-enforced max

_lzs = lzstring.LZString()


def decompress_gior(raw: bytes) -> list:
    # JS LZString.compressFromUint8Array splits each 16-bit char into two
    # big-endian bytes. Pair them back up and feed the resulting string to
    # plain decompress().
    s = "".join(chr((raw[i * 2] << 8) | raw[i * 2 + 1]) for i in range(len(raw) // 2))
    return json.loads(_lzs.decompress(s))


def user_exists(client: TrackedClient, username: str) -> bool:
    r = client.get(f"{API_BASE}/api/starsAndRanks", params={"u": username})
    return bool(r.json().get("ranks"))


def iter_user_replays(client: TrackedClient, username: str) -> Iterator[dict]:
    """Yield every replay listing for `username`, paging until the server returns []."""
    offset = 0
    while True:
        r = client.get(
            f"{API_BASE}/api/replaysForUsername",
            params={"u": username, "offset": offset, "count": PAGE_SIZE},
        )
        page = r.json()
        if not page:
            return
        yield from page
        offset += len(page)


def list_user_replays(client: TrackedClient, username: str, limit: int) -> list[dict]:
    return list(itertools.islice(iter_user_replays(client, username), limit))


def fetch_replay(client: TrackedClient, replay_id: str) -> tuple[bytes, list]:
    r = client.get(f"{S3_BASE}/{replay_id}.gior")
    return r.content, decompress_gior(r.content)
