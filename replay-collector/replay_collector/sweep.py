import logging
from dataclasses import dataclass, field

import httpx

from replay_collector import db, generals_api
from replay_collector.client import (
    DEFAULT_RATES,
    RateLimiter,
    TooManyFailures,
    TrackedClient,
    make_client,
)

log = logging.getLogger(__name__)

LOG_EVERY_N_PAGES = 10


@dataclass
class SweepStats:
    username: str
    user_exists: bool = True
    listings_walked: int = 0
    listings_new: int = 0          # upsert_listing returned True
    ffa_total: int = 0
    ffa_has_full: int = 0          # already have raw bytes
    ffa_metadata_only: int = 0     # raw IS NULL — Pass 2 budget contribution
    pages_fetched: int = 0
    stop_reason: str = ""          # exhausted | max_listings | user_not_found | error


@dataclass
class SweepRunStats:
    per_user: list[SweepStats] = field(default_factory=list)
    aborted: bool = False
    abort_reason: str = ""


def _log_progress(stats: SweepStats) -> None:
    log.info(
        "[%s] page %d: walked=%d ffa: total=%d has_full=%d metadata_only=%d",
        stats.username, stats.pages_fetched, stats.listings_walked,
        stats.ffa_total, stats.ffa_has_full, stats.ffa_metadata_only,
    )


def _log_summary(stats: SweepStats) -> None:
    log.info(
        "[%s] done: %d pages, walked=%d new_listings=%d "
        "ffa: total=%d has_full=%d metadata_only=%d (stop=%s)",
        stats.username, stats.pages_fetched, stats.listings_walked,
        stats.listings_new, stats.ffa_total, stats.ffa_has_full,
        stats.ffa_metadata_only, stats.stop_reason,
    )


def sweep_one(
    client: TrackedClient, username: str, max_listings: int
) -> SweepStats:
    """Walk every page of `username`'s replay listings, upserting each row.
    No .gior fetches. Stops when the API runs out of pages or `max_listings`
    is reached (a safety rail, not a target)."""
    stats = SweepStats(username=username)

    if not generals_api.user_exists(client, username):
        log.warning("user %r not found on generals.io; skipping", username)
        stats.user_exists = False
        stats.stop_reason = "user_not_found"
        return stats

    log.info("[%s] starting sweep", username)
    for page in generals_api.iter_user_replay_pages(client, username):
        stats.pages_fetched += 1
        for entry in page:
            stats.listings_walked += 1
            if db.upsert_listing(entry):
                stats.listings_new += 1
            if entry.get("ladder_id") == "ffa":
                stats.ffa_total += 1
                if db.has_full_data(entry["id"]):
                    stats.ffa_has_full += 1
                else:
                    stats.ffa_metadata_only += 1
            if stats.listings_walked >= max_listings:
                stats.stop_reason = "max_listings"
                _log_summary(stats)
                return stats
        if stats.pages_fetched % LOG_EVERY_N_PAGES == 0:
            _log_progress(stats)

    stats.stop_reason = "exhausted"
    _log_summary(stats)
    return stats


def sweep_many(
    usernames: list[str], max_listings: int, max_failures: int
) -> SweepRunStats:
    """Sweep every player's full listing history. Owns the shared httpx.Client +
    RateLimiter + TrackedClient. Aborts the run on TooManyFailures; per-user
    HTTP errors are logged and the run continues."""
    run = SweepRunStats()
    limiter = RateLimiter(DEFAULT_RATES)

    with make_client() as http:
        client = TrackedClient(http, limiter, max_failures=max_failures)
        try:
            for username in usernames:
                try:
                    run.per_user.append(sweep_one(client, username, max_listings))
                except TooManyFailures as e:
                    run.aborted = True
                    run.abort_reason = str(e)
                    log.error("aborting run: %s", e)
                    break
                except httpx.HTTPError as e:
                    log.warning("user %r aborted mid-run: %s", username, e)
                    run.per_user.append(
                        SweepStats(username=username, stop_reason="error")
                    )
        except KeyboardInterrupt:
            run.aborted = True
            run.abort_reason = "interrupted by user (SIGINT)"
            log.warning("interrupted by user; finished users=%d", len(run.per_user))

    log.info(
        "sweep complete: users=%d aborted=%s",
        len(run.per_user), run.aborted,
    )
    return run
