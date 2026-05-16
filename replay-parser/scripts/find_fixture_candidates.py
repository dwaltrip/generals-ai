"""Find fixture candidates for sim integration tests.

Scans a corpus sample, parses + sims each replay, computes per-scenario
feature scores, prints top-K candidates per scenario.

Scenarios (per discussion with user):
  - 3-way action: tile-region × time-bucket cells with >=3 distinct players
  - 2-on-1 general attack: moves into a general from >=2 attackers near capture
  - capture during surrender: capture event after captured player's first AFK
  - same-timestep capture chain (A->B->C in one step)
  - multi-surrender: >=3 distinct AFK'd players
  - city-battling: total ownership transitions on city tiles
  - run-of-the-mill: replays closest to median game length

Usage (from replay-parser/):
    uv run python scripts/find_fixture_candidates.py [--per-bucket N] [--top-k K]
"""
import argparse
import heapq
import sqlite3
import sys

import numpy as np

from replay_parser._collector.config import DB_PATH
from replay_parser._collector.wire import decode as decode_blob
from replay_parser._shared import is_vanilla_ffa
from replay_parser.decode import decode_wire_array
import sim_core

from _sweep_common import bucket_and_sample, fetch_candidates, log


PROGRESS_EVERY = 200
RANDOM_SEED = 42
REGION = 5           # tile region edge (3-way action)
T_BUCKET = 100       # timestep bucket size (3-way action)
GEN_ATTACK_WINDOW = 3


def feat_3way_action(
    own_all: np.ndarray, num_players: int, H: int, W: int
) -> tuple[int, int, str]:
    T = own_all.shape[0]
    n_ry, n_rx = H // REGION, W // REGION
    if n_ry == 0 or n_rx == 0:
        return 0, 0, ""
    Hp, Wp = n_ry * REGION, n_rx * REGION
    n_buckets = (T + T_BUCKET - 1) // T_BUCKET
    Tp = n_buckets * T_BUCKET

    own_2d = own_all.reshape(T, H, W)
    presence = np.zeros((num_players, n_buckets, n_ry, n_rx), dtype=bool)
    for p in range(num_players):
        pm = (
            (own_2d[:, :Hp, :Wp] == p)
            .reshape(T, n_ry, REGION, n_rx, REGION)
            .any(axis=(2, 4))
        )
        if T < Tp:
            pad = np.zeros((Tp - T, n_ry, n_rx), dtype=bool)
            pm = np.concatenate([pm, pad], axis=0)
        pm = pm.reshape(n_buckets, T_BUCKET, n_ry, n_rx).any(axis=1)
        presence[p] = pm

    distinct = presence.sum(axis=0)  # [n_buckets, n_ry, n_rx]
    cells_3way = int((distinct >= 3).sum())
    peak = int(distinct.max())
    if peak >= 3:
        b, ry, rx = np.unravel_index(int(distinct.argmax()), distinct.shape)
        detail = (
            f"peak={peak} @ t-bucket={b}(t~{b * T_BUCKET}) region=({ry},{rx})"
        )
    else:
        detail = f"peak={peak}"
    return cells_3way, peak, detail


def feat_2on1_general(state, replay, initial_generals) -> tuple[int, str]:
    moves = replay.moves
    if len(moves.timestep) == 0:
        return 0, ""
    m_ts = moves.timestep
    m_idx = moves.index
    m_dest = moves.dest

    best = 0
    best_detail = ""
    for ce in state.capture_events:
        gt = initial_generals[ce.captured]
        if gt < 0:
            continue
        lo = ce.timestep - GEN_ATTACK_WINDOW
        hi = ce.timestep
        i0 = int(np.searchsorted(m_ts, lo, side="left"))
        i1 = int(np.searchsorted(m_ts, hi, side="right"))
        if i1 <= i0:
            continue
        mask = (m_dest[i0:i1] == gt) & (m_idx[i0:i1] != ce.captured)
        attackers = set(int(x) for x in m_idx[i0:i1][mask].tolist())
        if len(attackers) > best:
            best = len(attackers)
            best_detail = (
                f"capture@t={ce.timestep} captured=p{ce.captured} "
                f"attackers={sorted(attackers)}"
            )
    return best, best_detail


def feat_capture_during_surrender(state, replay) -> tuple[int, str]:
    afks = replay.afks
    if len(afks.timestep) == 0:
        return 0, ""
    first_afk: dict[int, int] = {}
    for idx, t in zip(afks.index.tolist(), afks.timestep.tolist()):
        first_afk.setdefault(int(idx), int(t))
    count = 0
    first_detail = ""
    for ce in state.capture_events:
        afk_t = first_afk.get(ce.captured)
        if afk_t is not None and afk_t < ce.timestep:
            count += 1
            if not first_detail:
                first_detail = (
                    f"capture@t={ce.timestep} captured=p{ce.captured} "
                    f"surrendered@t={afk_t}"
                )
    return count, first_detail


def feat_same_t_chain(state) -> tuple[int, str]:
    """A->B->C chain at one timestep: captured(e1) == captor(e2). Score = the
    number of such overlap players. Disjoint same-t captures score 0."""
    if not state.capture_events:
        return 0, ""
    by_t: dict[int, list] = {}
    for ce in state.capture_events:
        by_t.setdefault(ce.timestep, []).append(ce)
    best_links = 0
    best_detail = ""
    for t, events in by_t.items():
        if len(events) < 2:
            continue
        captors = {ce.captor for ce in events}
        captured = {ce.captured for ce in events}
        links = len(captors & captured)
        if links > best_links:
            best_links = links
            best_detail = f"t={t} links={links}: " + ", ".join(
                f"p{ce.captor}->p{ce.captured}" for ce in events
            )
    return best_links, best_detail


def feat_multi_surrender(replay) -> int:
    if len(replay.afks.timestep) == 0:
        return 0
    return len(set(int(x) for x in replay.afks.index.tolist()))


def feat_city_battling(state, own_all: np.ndarray) -> tuple[int, str]:
    cities = state.cities
    if not cities:
        return 0, ""
    transitions = (own_all[1:] != own_all[:-1]).sum(axis=0)
    cities_arr = np.array(cities, dtype=np.int64)
    city_trans = transitions[cities_arr]
    top_n = np.argsort(-city_trans)[:3]
    detail = "top tiles: " + ", ".join(
        f"{int(cities_arr[i])}={int(city_trans[i])}" for i in top_n
    )
    return int(city_trans.sum()), detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        candidates = fetch_candidates(conn, min_version=15)
        log(f"  {len(candidates):,} candidates")
        _, sampled = bucket_and_sample(
            candidates, per_bucket=args.per_bucket, seed=RANDOM_SEED
        )
        log(f"  sampled {len(sampled):,} replays")

        keys = [
            "3way_action", "2on1_general", "capture_during_surrender",
            "samet_chain", "multi_surrender", "city_battling",
        ]
        heaps: dict[str, list[tuple[int, str, str]]] = {k: [] for k in keys}
        all_replays: list[tuple[str, int]] = []
        skipped = {"nonvanilla": 0, "overflow": 0, "parse_err": 0}

        def push(key: str, score: int, detail: str, rid: str) -> None:
            if score <= 0:
                return
            heap = heaps[key]
            if len(heap) < args.top_k:
                heapq.heappush(heap, (score, rid, detail))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, rid, detail))

        for i, (rid, _started, _ver) in enumerate(sampled):
            blob = conn.execute(
                "SELECT wire_data FROM replays WHERE id = ?", (rid,)
            ).fetchone()[0]
            try:
                wire = decode_blob(blob)
            except Exception:
                skipped["parse_err"] += 1
                continue
            if not is_vanilla_ffa(wire):
                skipped["nonvanilla"] += 1
                continue
            try:
                replay = decode_wire_array(wire)
                state = sim_core.simulate(replay)
            except Exception:
                skipped["overflow"] += 1
                continue

            s = replay.static
            # Bind once: getter clones the full snapshot list per access.
            snaps_own_list = state.snapshots_ownership
            own_all = np.stack(snaps_own_list, axis=0)
            T = own_all.shape[0]
            initial_generals = list(s.initial_generals)

            cells_3way, _peak, det_3way = feat_3way_action(
                own_all, state.num_players, s.map_height, s.map_width
            )
            atkrs, det_atk = feat_2on1_general(state, replay, initial_generals)
            cds_count, det_cds = feat_capture_during_surrender(state, replay)
            chain_max, det_chain = feat_same_t_chain(state)
            multi_s = feat_multi_surrender(replay)
            city_flips, det_city = feat_city_battling(state, own_all)

            all_replays.append((rid, T))

            push("3way_action", cells_3way, det_3way, rid)
            push("2on1_general", atkrs, det_atk, rid)
            push("capture_during_surrender", cds_count, det_cds, rid)
            push("samet_chain", chain_max, det_chain, rid)
            push("multi_surrender", multi_s, "", rid)
            push("city_battling", city_flips, det_city, rid)

            if (i + 1) % PROGRESS_EVERY == 0:
                log(f"  {i + 1}/{len(sampled)}")
    finally:
        conn.close()

    log("")
    log(f"Processed: {len(all_replays):,}  (skipped: {skipped})")
    log("")

    for key in keys:
        log(f"== {key} ==")
        for score, rid, det in sorted(heaps[key], reverse=True):
            log(f"  rid={rid:<12} score={score:<5} {det}")
        log("")

    if all_replays:
        Ts = sorted(t for _, t in all_replays)
        median_T = Ts[len(Ts) // 2]
        closest = sorted(all_replays, key=lambda r: abs(r[1] - median_T))[
            : args.top_k
        ]
        log(f"== run-of-the-mill (closest to median T={median_T}) ==")
        for rid, t in closest:
            log(f"  rid={rid:<12} T={t}")
        log("")

        p10_T = Ts[len(Ts) // 10]
        near_p10 = sorted(all_replays, key=lambda r: abs(r[1] - p10_T))[
            : args.top_k
        ]
        log(f"== short games (closest to p10 T={p10_T}) ==")
        for rid, t in near_p10:
            log(f"  rid={rid:<12} T={t}")
        log("")

        shortest = sorted(all_replays, key=lambda r: r[1])[: args.top_k]
        log("== very shortest games ==")
        for rid, t in shortest:
            log(f"  rid={rid:<12} T={t}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
