"""Corpus driver — fan a `multiprocessing.Pool` over filtered FFA replays
and write the sim + meta sidecar pair per game.

Two-stage filter pipeline:
  - SQL-level: `ladder_id='ffa' AND version >= 15 AND player_count BETWEEN 4 AND 8
    AND wire_data IS NOT NULL AND at-least-one-curated-player-was-in-the-game`.
    Narrows the candidate set before any worker touches a wire blob.
  - Wire-level: `replay_parser._shared.is_vanilla_ffa` — rejects modifier-tile
    games, custom maps, team modes, etc. Runs in the worker right after
    decompression so we drop bad games before paying the simulator cost.

Per-game flow inside a worker (see `_process_replay`):
  1. Skip if both output files already exist (cheap restart).
  2. SELECT wire_data + per-perspective placement rows.
  3. Decompress → wire-level filter.
  4. Decode → simulate.
  5. Intersect curated names with this game's slots; apply parse-time
     noise floor on each candidate's rolling stats.
  6. If any perspectives survive: write `<id>.npz` + `<id>.meta.npz`.
  7. Return a `GameResult` either way; the orchestrator streams these
     out and writes a `_skipped_*.csv` log.
"""
import csv
import datetime as dt
import multiprocessing as mp
import sqlite3
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TextIO

import sim_core

from replay_parser._collector.wire import decode as decompress
from replay_parser._shared import is_vanilla_ffa
from replay_parser.decode import decode_wire_array
from replay_parser.errors import ArmyOverflowError
from replay_parser.git_state import capture_git_version
from replay_parser.metadata import build_metadata
from replay_parser.output import write_metadata, write_sim_output
from replay_parser.rolling_stats import (
    RollingStat,
    RollingStatsResult,
    compute_rolling_stats,
)


# Per design §5.3 / §A.4.
DEFAULT_ROLLING_1ST_FLOOR = 0.125
DEFAULT_ROLLING_TOP3_FLOOR = 0.375
DEFAULT_MIN_PRIOR_GAMES = 50


@dataclass(frozen=True, slots=True)
class NoiseFloor:
    rolling_1st: float = DEFAULT_ROLLING_1ST_FLOOR
    rolling_top3: float = DEFAULT_ROLLING_TOP3_FLOOR
    min_prior_games: int = DEFAULT_MIN_PRIOR_GAMES


@dataclass(frozen=True, slots=True)
class DriverConfig:
    db_path: Path
    intermediate_dir: Path
    curated_names: tuple[str, ...]
    repo_root: Path
    noise_floor: NoiseFloor = field(default_factory=NoiseFloor)
    allow_dirty: bool = False
    log_every: int = 100


@dataclass(frozen=True, slots=True)
class GameResult:
    replay_id: str
    status: str   # "written" or "skip:<reason>"
    detail: str = ""


# --------------------------------------------------------------------------
# Worker globals (process-local; set up by `_worker_init`).
# --------------------------------------------------------------------------
_conn: Optional[sqlite3.Connection] = None
_rolling: Optional[RollingStatsResult] = None
_curated: Optional[frozenset[str]] = None
_config: Optional[DriverConfig] = None
_sim_core_version: Optional[str] = None


def _worker_init(
    db_path: str,
    rolling: RollingStatsResult,
    curated: frozenset[str],
    config: DriverConfig,
    sim_core_version: str,
) -> None:
    global _conn, _rolling, _curated, _config, _sim_core_version
    _conn = sqlite3.connect(db_path)
    _rolling = rolling
    _curated = curated
    _config = config
    _sim_core_version = sim_core_version


def _process_replay(replay_id: str) -> GameResult:
    assert (
        _conn is not None and _rolling is not None and _curated is not None
        and _config is not None and _sim_core_version is not None
    )

    shard_dir = _config.intermediate_dir / replay_id[:2]
    sim_path = shard_dir / f"{replay_id}.npz"
    meta_path = shard_dir / f"{replay_id}.meta.npz"
    if sim_path.exists() and meta_path.exists():
        return GameResult(replay_id, "skip:already-written")

    try:
        return _process_replay_inner(replay_id, shard_dir, sim_path, meta_path)
    except ArmyOverflowError as e:
        # Single-tile army stacks above i16 max — design-documented log+skip
        # case (replay-parser-design.md §3). Surfaced separately from the
        # generic exception bucket so the tally distinguishes "expected rare
        # failure mode" from "unexpected bug."
        return GameResult(replay_id, "skip:army-overflow", str(e))
    except Exception as e:  # noqa: BLE001 — per-game isolation, see module docstring
        return GameResult(
            replay_id, "skip:exception",
            f"{type(e).__name__}: {e} | {traceback.format_exc(limit=2).strip()}",
        )


def _process_replay_inner(
    replay_id: str,
    shard_dir: Path,
    sim_path: Path,
    meta_path: Path,
) -> GameResult:
    assert (
        _conn is not None and _rolling is not None and _curated is not None
        and _config is not None and _sim_core_version is not None
    )

    row = _conn.execute(
        "SELECT wire_data FROM replays WHERE id = ?", (replay_id,)
    ).fetchone()
    if row is None or row[0] is None:
        return GameResult(replay_id, "skip:no-wire-data")

    wire = decompress(row[0])
    if not is_vanilla_ffa(wire):
        return GameResult(replay_id, "skip:wire-filter")

    replay = decode_wire_array(wire)
    state = sim_core.simulate(replay)

    # Match curated names against this game's wire usernames (slot order).
    # Wire `usernames` is canonical for slot identity — `replay_players`
    # rows give us placement, but the join key is the username string.
    placement_rows = _conn.execute(
        "SELECT rp.position, p.name "
        "FROM replay_players rp "
        "JOIN players p ON rp.player_id = p.id "
        "WHERE rp.replay_id = ?",
        (replay_id,),
    ).fetchall()
    placement_by_name = {name: pos + 1 for pos, name in placement_rows}  # listing position is 0-indexed; placement is 1-indexed

    rolling_for_replay = _rolling.by_replay.get(replay_id, {})

    survivors: list[tuple[int, str, int, RollingStat]] = []  # (slot, name, placement, stat)
    for slot, name in enumerate(replay.static.usernames):
        if name not in _curated:
            continue
        stat = rolling_for_replay.get(name)
        if stat is None:
            continue
        if stat.prior_games_count < _config.noise_floor.min_prior_games:
            continue
        if stat.rolling_1st_rate <= _config.noise_floor.rolling_1st:
            continue
        if stat.rolling_top3_rate <= _config.noise_floor.rolling_top3:
            continue
        if name not in placement_by_name:
            # Wire and listing data disagree on which names participated.
            # Skip the game rather than guess.
            return GameResult(
                replay_id, "skip:listing-wire-name-mismatch",
                f"name={name!r} in wire usernames but not in replay_players",
            )
        survivors.append((slot, name, placement_by_name[name], stat))

    if not survivors:
        return GameResult(replay_id, "skip:no-curated-perspectives")

    perspective_player_ids = [s for s, _n, _p, _st in survivors]
    placement = [p for _s, _n, p, _st in survivors]
    rolling_1st = [st.rolling_1st_rate for _s, _n, _p, st in survivors]
    rolling_top3 = [st.rolling_top3_rate for _s, _n, _p, st in survivors]
    prior_counts = [st.prior_games_count for _s, _n, _p, st in survivors]

    meta = build_metadata(
        state, replay,
        perspective_player_ids=perspective_player_ids,
        placement=placement,
        sim_core_version=_sim_core_version,
        rolling_1st_rate=rolling_1st,
        rolling_top3_rate=rolling_top3,
        prior_games_count=prior_counts,
    )

    shard_dir.mkdir(parents=True, exist_ok=True)
    write_sim_output(state, replay, sim_path)
    write_metadata(meta, meta_path)
    return GameResult(replay_id, "written", f"perspectives={len(survivors)}")


# --------------------------------------------------------------------------
# Orchestration (parent process)
# --------------------------------------------------------------------------

def select_candidate_replay_ids(
    conn: sqlite3.Connection,
    curated_names: list[str],
    limit: Optional[int] = None,
) -> list[str]:
    """Candidate set: passes SQL-level filter AND has at least one curated
    player. Ordered by `started ASC` for deterministic re-runs."""
    placeholders = ",".join("?" * len(curated_names))
    sql = f"""
        SELECT DISTINCT r.id
        FROM replays r
        JOIN replay_players rp ON rp.replay_id = r.id
        JOIN players p ON rp.player_id = p.id
        WHERE r.ladder_id = 'ffa'
          AND r.version >= 15
          AND r.player_count BETWEEN 4 AND 8
          AND r.wire_data IS NOT NULL
          AND p.name IN ({placeholders})
        ORDER BY r.started ASC, r.id ASC
    """
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"
    return [row[0] for row in conn.execute(sql, curated_names)]


def run_corpus_driver(
    config: DriverConfig,
    workers: int,
    limit: Optional[int] = None,
) -> Path:
    """Top-level orchestrator. Returns the path to the skip-log CSV."""
    config.intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Capture git state up front — aborts if dirty (unless allow_dirty).
    sim_core_version = capture_git_version(
        config.repo_root, allow_dirty=config.allow_dirty,
    )

    print(f"Curated players: {len(config.curated_names)}")
    print(f"DB: {config.db_path}")
    print(f"Output: {config.intermediate_dir}")
    print(f"sim_core_version: {sim_core_version}")
    print(f"Workers: {workers}  Noise floor: {config.noise_floor}\n")

    conn = sqlite3.connect(config.db_path)
    try:
        t0 = time.perf_counter()
        print("Computing rolling stats…")
        rolling = compute_rolling_stats(conn, list(config.curated_names))
        print(f"  done in {time.perf_counter() - t0:.1f}s — "
              f"{sum(len(v) for v in rolling.by_player.values()):,} (replay, player) records.\n")

        print("Selecting candidate replays…")
        t0 = time.perf_counter()
        candidates = select_candidate_replay_ids(conn, list(config.curated_names), limit=limit)
        print(f"  {len(candidates):,} candidates in {time.perf_counter() - t0:.1f}s.\n")
    finally:
        conn.close()

    log_path, log_fh, writer = _open_skip_log(config.intermediate_dir)
    if not candidates:
        log_fh.close()
        print("No candidates — nothing to do.")
        return log_path
    curated = frozenset(config.curated_names)

    counts: dict[str, int] = {}
    t0 = time.perf_counter()
    try:
        with mp.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(str(config.db_path), rolling, curated, config, sim_core_version),
        ) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_process_replay, candidates, chunksize=8),
                start=1,
            ):
                counts[result.status] = counts.get(result.status, 0) + 1
                if result.status != "written":
                    writer.writerow([result.replay_id, result.status, result.detail])
                    log_fh.flush()
                if i % config.log_every == 0 or i == len(candidates):
                    rate = i / max(time.perf_counter() - t0, 1e-6)
                    print(
                        f"  [{i:>6}/{len(candidates)}] {rate:.1f} games/s  "
                        + " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                        flush=True,
                    )
    finally:
        log_fh.close()

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s "
          f"({len(candidates)/max(elapsed,1e-6):.1f} games/s, {workers} workers).")
    print("Final tally:")
    for k, v in sorted(counts.items()):
        print(f"  {k:<35} {v}")
    print(f"\nSkip log: {log_path}")
    return log_path


def _open_skip_log(intermediate_dir: Path) -> tuple[Path, TextIO, Any]:
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = intermediate_dir / f"_skipped_{stamp}.csv"
    fh = open(path, "w", encoding="utf-8", newline="")
    writer = csv.writer(fh)
    writer.writerow(["replay_id", "status", "detail"])
    return path, fh, writer
