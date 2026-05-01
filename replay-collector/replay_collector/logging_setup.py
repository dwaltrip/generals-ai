"""Two-file logging + streaming bucket-progress writer.

The collector writes two log files per run:

- condensed: high-level progress (intro/bucket/summary lines), with dots
  streamed live so a `tail -f` user sees progress mid-bucket. Drops httpx
  INFO and per-replay save records.
- verbose: superset (everything at INFO/DEBUG); bucket-end summary written
  whole as a single record. httpx noise mixed in is fine here.

The condensed log is special: stdlib logging can't open a line, write
fragments over time, and close it. So we route the condensed handler
through StreamWriter which owns the file directly. When a normal log
record (e.g. a TrackedClient HTTP-failure warning) fires while a streamed
line is open, the writer closes the line, emits the record, and lets the
next dot resume on a fresh indented line — no prefix repeat.
"""

import datetime as dt
import logging
from pathlib import Path

DOTS_PER_FETCHES = 20

# Records routed through StreamWriter that match these are dropped from the
# condensed log; they still flow to the verbose handler.
_CONDENSED_DROP_LOGGER_PREFIXES = ("httpx",)
_CONDENSED_DROP_MSG_SUBSTRINGS = ("saved id=",)
# This logger is verbose-only: BucketProgress emits a complete bucket-summary
# record here at end-of-bucket so the verbose log gets a single tidy line.
# The condensed log already has the streamed version.
_VERBOSE_ONLY_LOGGER = "replay_collector.bucket"

_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def _filename_timestamp(now: dt.datetime) -> str:
    return now.strftime("%Y.%m.%d-%H.%M.%S")


class StreamWriter:
    """Owns the condensed-log file. Tracks whether a streamed line is open
    so log records can break in cleanly without garbling dot output."""

    _RESUME_INDENT = "    "

    def __init__(self, path: Path, formatter: logging.Formatter):
        self._f = open(path, "a", buffering=1)
        self._fmt = formatter
        self._line_open = False

    def write_log_record(self, record: logging.LogRecord) -> None:
        if self._is_filtered(record):
            return
        if self._line_open:
            self._f.write("\n")
            self._line_open = False
        self._f.write(self._fmt.format(record) + "\n")
        self._f.flush()

    def open_line(self, prefix: str) -> None:
        if self._line_open:
            self._f.write("\n")
        self._f.write(prefix)
        self._f.flush()
        self._line_open = True

    def append(self, fragment: str) -> None:
        # If a log record interrupted us, resume on a fresh indented line —
        # alignment is intentionally relaxed; warnings/errors are rare and
        # warrant the visual break.
        if not self._line_open:
            self._f.write(self._RESUME_INDENT)
            self._line_open = True
        self._f.write(fragment)
        self._f.flush()

    def close_line(self) -> None:
        if self._line_open:
            self._f.write("\n")
            self._f.flush()
        self._line_open = False

    def now_prefix(self) -> str:
        """Format a timestamp+level prefix matching the formatter, for use in
        streamed lines that bypass the logging module."""
        rec = logging.LogRecord(
            name="", level=logging.INFO, pathname="", lineno=0,
            msg="", args=None, exc_info=None,
        )
        return f"{self._fmt.formatTime(rec)} INFO "

    def _is_filtered(self, record: logging.LogRecord) -> bool:
        if record.name == _VERBOSE_ONLY_LOGGER:
            return True
        if any(record.name.startswith(p) for p in _CONDENSED_DROP_LOGGER_PREFIXES):
            return True
        msg = record.getMessage()
        return any(s in msg for s in _CONDENSED_DROP_MSG_SUBSTRINGS)

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

    _verbose_log = logging.getLogger(_VERBOSE_ONLY_LOGGER)

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


def setup_logging(tmp_dir: Path) -> tuple[Path, Path, BucketProgress]:
    """Wire root logger to write a condensed + verbose log under tmp_dir.
    Returns the two paths and the BucketProgress runner.py uses."""
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

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(verbose_handler)
    root.addHandler(condensed_handler)

    return condensed_path, verbose_path, BucketProgress(writer)
