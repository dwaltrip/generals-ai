from dataclasses import dataclass
import logging

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

LOG_EVERY_N_FETCHES = 100


@dataclass
class FillStats:
    fetched: int = 0      # successful S3 fetch
    saved: int = 0        # successful DB save (wire_data + decoded fields)
    errors: int = 0       # any failure during fetch or save
    bytes_total: int = 0
    aborted: bool = False
    abort_reason: str = ""


def fill(
    work_rows: list[tuple[str, str, int]], max_failures: int
) -> FillStats:
    """Fetch and save the .gior bytes for each (replay_id, owner_name, started)
    row in `work_rows`. Errors are logged but do not update the row, so a
    failed replay stays wire_data IS NULL and gets retried on the next run.

    Owns the shared httpx.Client + RateLimiter + TrackedClient. Aborts the
    whole run on TooManyFailures."""
    stats = FillStats()
    total = len(work_rows)
    if total == 0:
        log.info("no pending fetches; nothing to do.")
        return stats

    log.info("fill starting: %d replays in this batch", total)
    limiter = RateLimiter(DEFAULT_RATES)

    with make_client() as http:
        client = TrackedClient(http, limiter, max_failures=max_failures)
        try:
            for i, (replay_id, _owner_name, _started) in enumerate(work_rows, 1):
                try:
                    raw, decoded = generals_api.fetch_replay(client, replay_id)
                except TooManyFailures as e:
                    stats.aborted = True
                    stats.abort_reason = str(e)
                    log.error("aborting fill run: %s", e)
                    break
                except httpx.HTTPError:
                    # TrackedClient already logged + counted toward the budget.
                    stats.errors += 1
                    continue
                except generals_api.ReplayDecodeError as e:
                    log.warning("decode error for replay_id=%s: %r", replay_id, e.__cause__)
                    stats.errors += 1
                    continue

                stats.fetched += 1
                try:
                    db.save_full_data(replay_id, decoded)
                except Exception:
                    log.exception("failed to save replay_id=%s", replay_id)
                    stats.errors += 1
                    continue

                stats.saved += 1
                stats.bytes_total += len(raw)
                if i % LOG_EVERY_N_FETCHES == 0:
                    log.info(
                        "[fetch %d/%d] last_id=%s bytes=%d (saved=%d errors=%d total_bytes=%s)",
                        i, total, replay_id, len(raw),
                        stats.saved, stats.errors, _human_bytes(stats.bytes_total),
                    )
        except KeyboardInterrupt:
            stats.aborted = True
            stats.abort_reason = "interrupted by user (SIGINT)"
            log.warning(
                "interrupted by user; saved=%d errors=%d bytes=%s",
                stats.saved, stats.errors, _human_bytes(stats.bytes_total),
            )

    log.info(
        "fill complete: fetched=%d saved=%d errors=%d bytes=%s aborted=%s",
        stats.fetched, stats.saved, stats.errors,
        _human_bytes(stats.bytes_total), stats.aborted,
    )
    return stats


def _human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f}MB"
    return f"{n / (1024 * 1024 * 1024):.2f}GB"
