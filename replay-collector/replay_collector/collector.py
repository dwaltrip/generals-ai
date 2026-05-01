import json
import logging

import httpx
import lzstring

from replay_collector import db
from replay_collector.client import RateLimiter, host_of, make_client

log = logging.getLogger(__name__)

API_BASE = "https://generals.io"
S3_BASE = "https://generalsio-replays-na.s3.amazonaws.com"
PAGE_SIZE = 200  # server-enforced max

DEFAULT_RATES = {
    host_of(API_BASE): 1.0,
    host_of(S3_BASE): 1.0,
}

_lzs = lzstring.LZString()


def decompress_gior(raw: bytes) -> list:
    # JS LZString.compressFromUint8Array splits each 16-bit char into two
    # big-endian bytes. Pair them back up and feed the resulting string to
    # plain decompress().
    s = "".join(chr((raw[i * 2] << 8) | raw[i * 2 + 1]) for i in range(len(raw) // 2))
    return json.loads(_lzs.decompress(s))


def user_exists(client: httpx.Client, limiter: RateLimiter, username: str) -> bool:
    url = f"{API_BASE}/api/starsAndRanks"
    limiter.acquire(host_of(url))
    r = client.get(url, params={"u": username})
    if r.status_code != 200:
        return False
    data = r.json()
    return bool(data.get("ranks"))


def list_user_replays(
    client: httpx.Client,
    limiter: RateLimiter,
    username: str,
    limit: int,
) -> list[dict]:
    """Page /api/replaysForUsername until we hit `limit` or server returns []."""
    url = f"{API_BASE}/api/replaysForUsername"
    host = host_of(url)
    out: list[dict] = []
    offset = 0
    while len(out) < limit:
        page_size = min(PAGE_SIZE, limit - len(out))
        limiter.acquire(host)
        r = client.get(url, params={"u": username, "offset": offset, "count": page_size})
        r.raise_for_status()
        page = r.json()
        if not page:
            break
        out.extend(page)
        offset += len(page)
    return out


def fetch_replay(client: httpx.Client, limiter: RateLimiter, replay_id: str) -> tuple[bytes, list]:
    url = f"{S3_BASE}/{replay_id}.gior"
    limiter.acquire(host_of(url))
    r = client.get(url)
    r.raise_for_status()
    raw = r.content
    return raw, decompress_gior(raw)


def collect(username: str, limit: int) -> None:
    limiter = RateLimiter(DEFAULT_RATES)
    with make_client() as client:
        if not user_exists(client, limiter, username):
            log.error("user %r not found on generals.io; aborting", username)
            return

        meta = list_user_replays(client, limiter, username, limit)
        log.info("found %d replay(s) for %r", len(meta), username)

        for entry in meta:
            replay_id = entry["id"]
            if db.has_replay(replay_id):
                log.info("skip id=%s (already in db)", replay_id)
                continue
            try:
                raw, decoded = fetch_replay(client, limiter, replay_id)
            except httpx.HTTPError as e:
                log.warning("fetch failed for %s: %s", replay_id, e)
                continue
            db.save_replay(entry, decoded, raw)
            log.info(
                "saved id=%s type=%s turns=%d bytes=%d",
                replay_id,
                entry.get("type"),
                entry.get("turns"),
                len(raw),
            )
