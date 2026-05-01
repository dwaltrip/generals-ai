import time
from urllib.parse import urlparse

import httpx

# TODO: consider adding contact email to UA (e.g. "(+contact: <email>)") so
# the generals.io operators can reach us if our traffic ever causes friction.
USER_AGENT = "generals-ai-replay-collector/0.1"
DEFAULT_TIMEOUT = 30.0


class RateLimiter:
    """Per-host minimum-interval limiter, single-threaded.

    Semantics: at least `1 / rate` seconds between successive `acquire()`
    returns for the same host. Hosts not configured are unrestricted.
    """

    def __init__(self, rates_per_second: dict[str, float]):
        self._min_intervals = {h: 1.0 / r for h, r in rates_per_second.items()}
        self._last_start: dict[str, float] = {}

    def acquire(self, host: str) -> None:
        interval = self._min_intervals.get(host)
        if interval is None:
            return
        now = time.monotonic()
        wait = self._last_start.get(host, 0.0) + interval - now
        if wait > 0:
            time.sleep(wait)
        self._last_start[host] = time.monotonic()


def make_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=DEFAULT_TIMEOUT,
        follow_redirects=True,
    )


def host_of(url: str) -> str:
    return urlparse(url).hostname or ""
