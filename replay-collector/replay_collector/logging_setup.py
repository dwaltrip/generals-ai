"""Two-file logging + streaming bucket-progress writer.

The collector writes two log files per run:

- condensed: high-level progress (intro/bucket/summary lines), with dots
  streamed live so a `tail -f` user sees progress mid-bucket.
- verbose: superset (everything at INFO/DEBUG); bucket-end summary written
  whole as a single record. httpx noise mixed in is fine here.

Routing:

- "Verbose-only" event categories (`httpx`, `replay_collector.bucket`,
  `replay_collector.saved`) attach the verbose handler directly and set
  `propagate = False` so they never reach root → the condensed handler.
- Everything else flows through root → both handlers.

Naming convention: loggers under `replay_collector.*` that follow the
module path (`replay_collector.runner`, `.client`) are module loggers.
Loggers named for an event category (`replay_collector.bucket`,
`.saved`) are intentionally not module-tied — the name is the routing
handle.

The condensed log is special: stdlib logging can't open a line, write
fragments over time, and close it. So the condensed handler routes
through StreamWriter which owns the file directly. When a normal log
record (e.g. a TrackedClient HTTP-failure warning) fires while a streamed
line is open, the writer closes the line, emits the record, and lets the
next dot resume on a fresh indented line — no prefix repeat.

StreamWriter also tees every write to stdout, so the running terminal
sees the same content as the condensed log (after the path lines that
__main__ prints before logging is wired up).
"""

import atexit
import datetime as dt
import logging
import sys
from pathlib import Path

DOTS_PER_FETCHES = 20

_VERBOSE_ONLY_LOGGERS = (
    "httpx",
    "replay_collector.bucket",
    "replay_collector.saved",
)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"

_configured: bool = False
_cached_result: "tuple[Path, Path, BucketProgress] | None" = None


def _filename_timestamp(now: dt.datetime) -> str:
    return now.strftime("%Y.%m.%d-%H.%M.%S")


class StreamWriter:
    """Owns the condensed-log file and tees every write to stdout. Tracks
    whether a streamed line is open so log records can break in cleanly
    without garbling dot output."""

    _RESUME_INDENT = "    "

    def __init__(self, path: Path, formatter: logging.Formatter):
        self._f = open(path, "a", buffering=1)
        self._fmt = formatter
        self._line_open = False

    def _write(self, s: str) -> None:
        self._f.write(s)
        sys.stdout.write(s)

    def _flush(self) -> None:
        self._f.flush()
        sys.stdout.flush()

    def write_log_record(self, record: logging.LogRecord) -> None:
        if self._line_open:
            self._write("\n")
            self._line_open = False
        self._write(self._fmt.format(record) + "\n")
        self._flush()

    def open_line(self, prefix: str) -> None:
        if self._line_open:
            self._write("\n")
        self._write(prefix)
        self._flush()
        self._line_open = True

    def append(self, fragment: str) -> None:
        # If a log record interrupted us, resume on a fresh indented line —
        # alignment is intentionally relaxed; warnings/errors are rare and
        # warrant the visual break.
        if not self._line_open:
            self._write(self._RESUME_INDENT)
            self._line_open = True
        self._write(fragment)
        self._flush()

    def close_line(self) -> None:
        if self._line_open:
            self._write("\n")
            self._flush()
        self._line_open = False

    def now_prefix(self) -> str:
        # Build the prefix by formatting an empty-message record, so any
        # change to _LOG_FORMAT carries through. Assumes the format ends
        # with %(message)s — true today; no guard.
        rec = logging.LogRecord(
            name="", level=logging.INFO, pathname="", lineno=0,
            msg="", args=None, exc_info=None,
        )
        return self._fmt.format(rec)

    def close(self) -> None:
        self.close_line()
        self._f.close()


class StreamWriterHandler(logging.Handler):
    """logging.Handler delegate so the writer can serialize streamed dot
    output with normal log records."""

    def __init__(self, writer: StreamWriter):
        super().__init__()
        self.writer = writer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.writer.write_log_record(record)
        except Exception:
            self.handleError(record)


class BucketProgress:
    """Per-user, per-bucket progress state. Streams the bucket prefix + dots
    to the condensed log via StreamWriter; emits a complete bucket-summary
    record to the verbose log at end-of-bucket."""

    _verbose_log = logging.getLogger("replay_collector.bucket")

    def __init__(self, writer: StreamWriter):
        self._writer = writer
        self._bucket_idx = 0
        self._fetches = 0
        self._dots_emitted = 0
        # Captured at start_bucket so end_bucket's verbose record matches.
        self._n_listings = 0
        self._ffa_count = 0
        self._cached = 0
        self._to_fetch = 0

    def start_user(self) -> None:
        self._bucket_idx = 0

    def start_bucket(
        self, n_listings: int, ffa_count: int, cached: int, to_fetch: int
    ) -> None:
        self._bucket_idx += 1
        self._fetches = 0
        self._dots_emitted = 0
        self._n_listings = n_listings
        self._ffa_count = ffa_count
        self._cached = cached
        self._to_fetch = to_fetch
        prefix = (
            f"{self._writer.now_prefix()}"
            f"    bucket {self._bucket_idx:2d}: "
            f"{n_listings:3d} listings, {ffa_count:3d} FFA "
            f"({cached:3d} cached, {to_fetch:3d} to fetch) "
        )
        self._writer.open_line(prefix)

    def fetch_done(self) -> None:
        self._fetches += 1
        if self._fetches % DOTS_PER_FETCHES == 0:
            self._writer.append(".")
            self._dots_emitted += 1

    def end_bucket(self) -> None:
        # Ceil-at-end: any nonzero remainder gets one final dot.
        if self._fetches % DOTS_PER_FETCHES != 0:
            self._writer.append(".")
            self._dots_emitted += 1
        self._writer.close_line()
        dots = "." * self._dots_emitted
        self._verbose_log.info(
            "    bucket %2d: %3d listings, %3d FFA (%3d cached, %3d to fetch) %s",
            self._bucket_idx, self._n_listings, self._ffa_count,
            self._cached, self._to_fetch, dots,
        )


_simple_configured: bool = False
_simple_log_path: Path | None = None


def setup_simple_logging(tmp_dir: Path, name: str) -> Path:
    """Single-file logging for runs that don't need the condensed/verbose
    split or bucket-progress streaming. INFO+ to a timestamped log file
    under tmp_dir AND to stdout. httpx noise is silenced to WARNING.

    `name` is embedded in the log filename (e.g. "sweep_metadata",
    "fetch_gior")."""
    global _simple_configured, _simple_log_path
    if _simple_configured:
        logging.getLogger(__name__).warning(
            "setup_simple_logging() called more than once; returning existing path."
        )
        return _simple_log_path

    tmp_dir.mkdir(parents=True, exist_ok=True)
    ts = _filename_timestamp(dt.datetime.now())
    log_path = tmp_dir / f"{ts}-{name}.log"

    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)

    _simple_configured = True
    _simple_log_path = log_path
    return log_path


def setup_logging(tmp_dir: Path) -> tuple[Path, Path, BucketProgress]:
    """Wire root logger to write a condensed + verbose log under tmp_dir.
    Returns the two paths and the BucketProgress runner.py uses."""
    global _configured, _cached_result
    if _configured:
        logging.getLogger(__name__).warning(
            "setup_logging() called more than once; returning existing handlers."
        )
        return _cached_result

    tmp_dir.mkdir(parents=True, exist_ok=True)
    ts = _filename_timestamp(dt.datetime.now())
    condensed_path = tmp_dir / f"{ts}-replay_collector.log"
    verbose_path = tmp_dir / f"{ts}-replay_collector-verbose.log"

    formatter = logging.Formatter(_LOG_FORMAT)

    verbose_handler = logging.FileHandler(verbose_path)
    verbose_handler.setLevel(logging.INFO)
    verbose_handler.setFormatter(formatter)

    writer = StreamWriter(condensed_path, formatter)
    condensed_handler = StreamWriterHandler(writer)
    condensed_handler.setLevel(logging.INFO)
    atexit.register(writer.close)

    # Verbose-only categories: attach the verbose handler directly and stop
    # propagation, so the condensed handler on root never sees these.
    for name in _VERBOSE_ONLY_LOGGERS:
        logger = logging.getLogger(name)
        logger.addHandler(verbose_handler)
        logger.propagate = False
        logger.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(verbose_handler)
    root.addHandler(condensed_handler)

    progress = BucketProgress(writer)
    _configured = True
    _cached_result = (condensed_path, verbose_path, progress)
    return _cached_result
