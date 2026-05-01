import logging
from dataclasses import dataclass, field

import httpx

from replay_collector import db, generals_api
from replay_collector.client import (
    RateLimiter,
    TooManyFailures,
    TrackedClient,
    host_of,
    make_client,
)

log = logging.getLogger(__name__)

DEFAULT_RATES = {
    host_of(generals_api.API_BASE): 1.0,
    host_of(generals_api.S3_BASE): 1.0,
}

# Listing metadata is upserted for every replay we walk past; the .gior bytes
# are only fetched (and decoded fields populated) for games whose ladder_id is
# in this set. Widen later for a richer corpus.
FULL_DATA_LADDER_ID_FILTER = {"ffa"}

DEFAULT_MAX_LISTINGS_PER_USER = 1000
DEFAULT_MAX_FAILURES = 10


@dataclass
class UserStats:
    username: str
    user_exists: bool = True
    listings_walked: int = 0
    ffa_found: int = 0
    new_listings: int = 0
    full_data_fetched: int = 0
    full_data_already_had: int = 0
    fetch_errors: int = 0
    stop_reason: str = ""  # target_reached | max_listings | exhausted | user_not_found


@dataclass
class RunStats:
    per_user: list[UserStats] = field(default_factory=list)
    aborted: bool = False
    abort_reason: str = ""

    def totals(self) -> dict[str, int]:
        keys = (
            "listings_walked", "ffa_found", "new_listings",
            "full_data_fetched", "full_data_already_had", "fetch_errors",
        )
        return {k: sum(getattr(u, k) for u in self.per_user) for k in keys}


def collect_one(
    client: TrackedClient, username: str, n_ffa: int, max_listings: int
) -> UserStats:
    """Walk `username`'s recent replays, persisting every listing as metadata
    and downloading the .gior for FFA games until we hit `n_ffa` FFAs found,
    walk past `max_listings` total entries, or run out of pages."""
    stats = UserStats(username=username)

    if not generals_api.user_exists(client, username):
        log.warning("user %r not found on generals.io; skipping", username)
        stats.user_exists = False
        stats.stop_reason = "user_not_found"
        return stats

    for entry in generals_api.iter_user_replays(client, username):
        stats.listings_walked += 1
        if db.upsert_listing(entry):
            stats.new_listings += 1

        if entry.get("ladder_id") in FULL_DATA_LADDER_ID_FILTER:
            stats.ffa_found += 1
            replay_id = entry["id"]
            if db.has_full_data(replay_id):
                stats.full_data_already_had += 1
            else:
                try:
                    raw, decoded = generals_api.fetch_replay(client, replay_id)
                except httpx.HTTPError:
                    # TrackedClient already logged + counted toward the budget.
                    stats.fetch_errors += 1
                else:
                    db.save_full_data(replay_id, raw, decoded)
                    stats.full_data_fetched += 1
                    log.info(
                        "saved id=%s type=%s turns=%d bytes=%d",
                        replay_id, entry.get("type"), entry.get("turns"), len(raw),
                    )
            if stats.ffa_found >= n_ffa:
                stats.stop_reason = "target_reached"
                break

        if stats.listings_walked >= max_listings:
            stats.stop_reason = "max_listings"
            break
    else:
        stats.stop_reason = "exhausted"

    log.info(
        "user=%r walked=%d ffa_found=%d new_listings=%d fetched=%d "
        "already_had=%d fetch_errors=%d stop=%s",
        username, stats.listings_walked, stats.ffa_found, stats.new_listings,
        stats.full_data_fetched, stats.full_data_already_had,
        stats.fetch_errors, stats.stop_reason,
    )
    return stats


def collect_many(
    usernames: list[str],
    n_ffa: int,
    max_listings: int = DEFAULT_MAX_LISTINGS_PER_USER,
    max_failures: int = DEFAULT_MAX_FAILURES,
) -> RunStats:
    """Collect the N most recent FFA replays for each username.

    Owns the shared httpx.Client + RateLimiter + TrackedClient for the run.
    Aborts the whole run if the failure budget is exhausted; otherwise a
    single user's errors don't block the rest.
    """
    run = RunStats()
    limiter = RateLimiter(DEFAULT_RATES)

    with make_client() as http:
        client = TrackedClient(http, limiter, max_failures=max_failures)
        for username in usernames:
            try:
                run.per_user.append(collect_one(client, username, n_ffa, max_listings))
            except TooManyFailures as e:
                run.aborted = True
                run.abort_reason = str(e)
                log.error("aborting run: %s", e)
                break
            except httpx.HTTPError as e:
                # Mid-user HTTP error that didn't trip the budget: TrackedClient
                # already logged + counted it. Record a partial UserStats so the
                # run summary reflects which user broke, then move on.
                log.warning("user %r aborted mid-run: %s", username, e)
                run.per_user.append(
                    UserStats(username=username, stop_reason="error")
                )

    totals = run.totals()
    log.info(
        "run complete: users=%d aborted=%s | walked=%d ffa_found=%d "
        "new_listings=%d fetched=%d already_had=%d fetch_errors=%d failures=%d",
        len(run.per_user), run.aborted,
        totals["listings_walked"], totals["ffa_found"], totals["new_listings"],
        totals["full_data_fetched"], totals["full_data_already_had"],
        totals["fetch_errors"], client.failures,
    )
    return run
