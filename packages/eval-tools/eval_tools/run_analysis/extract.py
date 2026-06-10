"""Reduce one GameRecord to flat analysis rows.

Two row kinds, written as CSVs by the CLI:

    player rows — one per (game, seat): outcome, kills, behavioral scalars,
        whole-game share summaries, policy/value diagnostics.
    stage rows — one per (game, stage, living seat): economy share/rank
        snapshots on a stage axis. The only axis implemented is 'alive'
        (the spans between eliminations); a tick-grid axis can be added by
        generating different StageSpan lists and reusing snapshot_rows.

Share/rank conventions: shares are fractions of the player-owned total
(neutral/mountain excluded); ranks are 1-based among living players with
ties sharing the best rank; a "leader" flag requires a strict maximum.
Mid-span snapshots (rather than span entry) avoid the mechanical share
spike from the capture-transfer that opens each span.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval_tools.run_analysis.loader import GameRecord, Group


@dataclass
class StageSpan:
    """Snapshot range [t_start, t_end] (inclusive) for one stage of a game."""

    axis: str  # 'alive'
    stage: int  # players alive during the span
    t_start: int
    t_end: int

    @property
    def t_mid(self) -> int:
        return (self.t_start + self.t_end) // 2


def alive_spans(rec: GameRecord) -> list[StageSpan]:
    """Stage spans between eliminations, using the snapshot convention
    (death at tick d → last alive snapshot d)."""
    spans = []
    t_prev = 0
    alive = rec.num_players
    for tick, _seat in rec.death_events.tolist():
        if tick >= t_prev:
            spans.append(StageSpan("alive", alive, t_prev, tick))
        t_prev = tick + 1
        alive -= 1
    if alive >= 2 and t_prev <= rec.game_length:
        # Draw: the run hit max_turns with 2+ players still alive.
        spans.append(StageSpan("alive", alive, t_prev, rec.game_length))
    return spans


def tick_spans(rec: GameRecord, ticks: range) -> list[StageSpan]:
    """Degenerate one-snapshot spans on a fixed tick grid (axis='tick')."""
    return [StageSpan("tick", t, t, t) for t in ticks if t <= rec.game_length]


def per_tick_series(rec: GameRecord) -> dict[str, np.ndarray]:
    """Per-snapshot land/army/city counts, each [T+1, P]."""
    P = rec.num_players
    own3 = rec.ownership[:, None, :] == np.arange(P, dtype=np.int8)[None, :, None]
    land = own3.sum(axis=2)
    army = (rec.armies[:, None, :] * own3).sum(axis=2, dtype=np.int64)

    ticks = np.arange(rec.ownership.shape[0])
    present = ticks[:, None] >= rec.cities_present_at[None, :]  # [T+1, C]
    city_owner = rec.ownership[:, rec.cities.astype(np.int64)]  # [T+1, C]
    cities = np.stack(
        [((city_owner == p) & present).sum(axis=1) for p in range(P)], axis=1
    )
    return {"land": land, "army": army, "cities": cities}


def placements(rec: GameRecord) -> list[float]:
    """Final placement per seat, 1 = winner. Eliminated seats rank by death
    order (first death places last). Draw survivors tie-share the top places."""
    P = rec.num_players
    out = [0.0] * P
    place = P
    for _tick, seat in rec.death_events.tolist():
        out[seat] = float(place)
        place -= 1
    survivors = [s for s in range(P) if out[s] == 0.0]
    if rec.winner_seat is not None:
        assert survivors == [rec.winner_seat]
        out[rec.winner_seat] = 1.0
    else:
        tied = sum(range(1, len(survivors) + 1)) / len(survivors)
        for s in survivors:
            out[s] = tied
    return out


def _ranks(values: np.ndarray, alive: list[bool]) -> list[float]:
    """1-based ranks among living players; ties share the best rank; dead → nan."""
    out = [float("nan")] * len(values)
    for i, v in enumerate(values):
        if alive[i]:
            out[i] = 1.0 + sum(
                1 for j, w in enumerate(values) if alive[j] and w > v
            )
    return out


def _shares(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts / total if total > 0 else np.zeros_like(counts, dtype=float)


def snapshot_rows(
    rec: GameRecord,
    spans: list[StageSpan],
    series: dict[str, np.ndarray],
    group_of_seat: list[str],
    placement: list[float],
) -> list[dict]:
    """One row per (span, living seat): mid-span shares/ranks + span-mean shares."""
    rows = []
    death_tick = {seat: tick for tick, seat in rec.death_events.tolist()}
    for span in spans:
        alive = [death_tick.get(s, rec.game_length) >= span.t_start for s in range(rec.num_players)]
        mid = {k: series[k][span.t_mid] for k in series}
        shares_mid = {k: _shares(v) for k, v in mid.items()}
        ranks_mid = {k: _ranks(v, alive) for k, v in mid.items()}
        span_mean = {
            k: _shares(series[k][span.t_start : span.t_end + 1].sum(axis=0))
            for k in series
        }
        for s in range(rec.num_players):
            if not alive[s]:
                continue
            row: dict = {
                "game_id": rec.game_id,
                "replay_id": rec.replay_id,
                "seat": s,
                "group": group_of_seat[s],
                "axis": span.axis,
                "stage": span.stage,
                "span_start": span.t_start,
                "span_end": span.t_end,
                "t_mid": span.t_mid,
                "eventual_win": int(rec.winner_seat == s),
                "placement": placement[s],
                "is_draw_game": int(rec.winner_seat is None),
            }
            for k in series:
                row[f"{k}_share_mid"] = float(shares_mid[k][s])
                row[f"{k}_rank_mid"] = ranks_mid[k][s]
                row[f"{k}_share_span"] = float(span_mean[k][s])
                strict_max = mid[k][s] > 0 and all(
                    mid[k][j] < mid[k][s]
                    for j in range(rec.num_players)
                    if j != s and alive[j]
                )
                row[f"{k}_leader_mid"] = int(strict_max)
            rows.append(row)
    return rows


def player_rows(
    rec: GameRecord,
    series: dict[str, np.ndarray],
    group_of_seat: list[str],
    placement: list[float],
) -> list[dict]:
    P = rec.num_players
    death_tick = {seat: tick for tick, seat in rec.death_events.tolist()}
    kills = [0] * P
    killed_by = [-1] * P
    for _tick, captor, victim in rec.capture_events.tolist():
        kills[captor] += 1
        killed_by[victim] = captor

    move_analysis = rec.metrics.get("move_analysis", [{}] * P)
    diags = {d["player"]: d for d in rec.metrics.get("policy_diagnostics", [])}

    rows = []
    for s in range(P):
        last_alive = death_tick.get(s, rec.game_length)
        # Whole-life share summaries (snapshots while the seat was alive).
        alive_slice = slice(0, last_alive + 1)
        army_alive = series["army"][alive_slice]
        land_alive = series["land"][alive_slice]
        army_share = army_alive[:, s] / army_alive.sum(axis=1)
        land_share = land_alive[:, s] / land_alive.sum(axis=1)
        army_rank1 = (
            army_alive[:, s][:, None] >= army_alive
        ).all(axis=1)  # weak rank-1: no one strictly larger

        ma = move_analysis[s] if s < len(move_analysis) else {}
        moves = ma.get("total_moves", 0)
        passes = ma.get("total_passes", 0)
        d = diags.get(s, {})
        row = {
            "game_id": rec.game_id,
            "game_index": rec.game_index,
            "replay_id": rec.replay_id,
            "seat": s,
            "policy_idx": rec.slot_map[s],
            "group": group_of_seat[s],
            "won": int(rec.winner_seat == s),
            "placement": placement[s],
            "is_draw_game": int(rec.winner_seat is None),
            "game_length": rec.game_length,
            "death_tick": death_tick.get(s, -1),
            "survival_frac": last_alive / rec.game_length,
            "kills": kills[s],
            "killed_by_seat": killed_by[s],
            "final_land": rec.player_stats[s]["land"],
            "final_army": rec.player_stats[s]["army"],
            "mean_army_share_alive": float(army_share.mean()),
            "mean_land_share_alive": float(land_share.mean()),
            "frac_ticks_army_rank1": float(army_rank1.mean()),
            # behavioral fingerprints (from the in-run collectors)
            "total_moves": moves,
            "total_passes": passes,
            "pass_rate": passes / (moves + passes) if moves + passes else float("nan"),
            "ones_traversed": ma.get("ones_traversed", 0),
            "ones_frac": ma.get("ones_traversed", 0) / moves if moves else float("nan"),
            "attack_move_count": ma.get("attack_move_count", 0),
            "attack_chain_count": ma.get("attack_chain_count", 0),
            "attack_chain_avg_length": ma.get("attack_chain_avg_length", float("nan")),
            "attack_chain_max_length": ma.get("attack_chain_max_length", float("nan")),
            "attack_army_avg": ma.get("attack_army_avg", float("nan")),
            "attack_army_p50": ma.get("attack_army_p50", float("nan")),
            "attack_army_p90": ma.get("attack_army_p90", float("nan")),
        }
        # policy/value diagnostics (rides along for later value-head analysis)
        policy = d.get("policy", {})
        value = d.get("value", {})
        nan = float("nan")
        exp_placement = value.get("exp_placement", {})
        row.update(
            {
                "policy_entropy_mean": policy.get("entropy", {}).get("mean", nan),
                "policy_top1_mean": policy.get("top1_prob", {}).get("mean", nan),
                "pass_prob_mean": d.get("pass_prob", {}).get("mean", nan),
                "value_exp_placement_mean": exp_placement.get("mean", nan),
                "value_exp_placement_std": exp_placement.get("std", nan),
                "n_no_legal": d.get("decisions", {}).get("n_no_legal", 0),
            }
        )
        rows.append(row)
    return rows


def extract_game(
    rec: GameRecord,
    groups: list[Group],
    early_grid: range = range(25, 151, 25),
) -> tuple[list[dict], list[dict]]:
    """All rows for one game: (player_rows, stage_rows).

    Stage rows cover the alive-count axis plus an early-game tick grid
    (the latter feeds the early land-share curve)."""
    label_of_policy = {
        i: g.label for g in groups for i in g.policy_indices
    }
    group_of_seat = [label_of_policy[p] for p in rec.slot_map]
    series = per_tick_series(rec)
    placement = placements(rec)
    spans = alive_spans(rec) + tick_spans(rec, early_grid)
    return (
        player_rows(rec, series, group_of_seat, placement),
        snapshot_rows(rec, spans, series, group_of_seat, placement),
    )
