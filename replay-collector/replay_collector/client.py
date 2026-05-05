import logging
import time
from urllib.parse import urlparse

import httpx

from replay_collector.config import API_BASE, S3_BASE, config

DEFAULT_TIMEOUT = 30.0
USER_AGENT_BASE = "generals-ai-replay-collector/0.1"
# Add contact email so generals.io operators can reach us if we're causing issues.
USER_AGENT = f"{USER_AGENT_BASE} (+mailto:{config.UA_CONTACT_EMAIL})"

log = logging.getLogger(__name__)


def host_of(url: str) -> str:
    return urlparse(url).hostname or ""


# Per-host request budget shared across all runners. generals.io's API gets
# the conservative 1/sec the community tolerates for hobby projects; the S3
# replay bucket is Amazon-hosted archival storage, so we pace it faster.
DEFAULT_RATES = {
    host_of(API_BASE): 1.0,
    host_of(S3_BASE): 2.0,
}


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


class TooManyFailures(Exception):
    """Raised by TrackedClient when the failure budget is exhausted."""

    def __init__(self, count: int):
        super().__init__(f"failure budget exhausted after {count} failed request(s)")
        self.count = count


class TrackedClient:
    """httpx.Client + per-host rate limit + run-wide failure budget.

    Bundles the three concerns the collector cares about into one call site:
    every `.get()` acquires the host limiter, raises on 4xx/5xx, and counts
    HTTP errors against a shared budget. When the budget is exhausted the
    next failure raises TooManyFailures so the caller can abort the run.
    """

    _BODY_SNIPPET_LIMIT = 500

    def __init__(self, client: httpx.Client, limiter: RateLimiter, max_failures: int):
        self._client = client
        self._limiter = limiter
        self._max_failures = max_failures
        self._failures = 0

    @property
    def failures(self) -> int:
        return self._failures

    def get(self, url: str, **kwargs) -> httpx.Response:
        self._limiter.acquire(host_of(url))
        try:
            r = self._client.get(url, **kwargs)
            r.raise_for_status()
            return r
        except httpx.HTTPError as e:
            self._failures += 1
            self._log_failure(url, e)
            if self._failures >= self._max_failures:
                raise TooManyFailures(self._failures) from e
            raise

    def _log_failure(self, url: str, exc: httpx.HTTPError) -> None:
        if isinstance(exc, httpx.HTTPStatusError):
            body = exc.response.text[: self._BODY_SNIPPET_LIMIT]
            log.warning(
                "request failed [%d/%d]: %s -> %d %s; body=%r",
                self._failures, self._max_failures,
                url, exc.response.status_code, exc.response.reason_phrase, body,
            )
        else:
            log.warning(
                "request failed [%d/%d]: %s -> %s: %s",
                self._failures, self._max_failures,
                url, type(exc).__name__, exc,
            )
