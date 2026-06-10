"""Turn extracted rows into report-ready tables.

Every function takes the flat row lists (player rows, stage rows) and the
group labels; nothing here re-reads game files. The combined output dict is
what lands in analysis/metrics.json.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math

from eval_tools.run_analysis import stats


ECONOMY_KEYS = ("army", "land", "cities")

FINGERPRINT_METRICS = (
    "pass_rate",
    "ones_frac",
    "attack_move_count",
    "attack_chain_count",
    "attack_chain_avg_length",
    "attack_chain_max_length",
    "attack_army_p50",
    "attack_army_p90",
    "policy_entropy_mean",
    "policy_top1_mean",
    "pass_prob_mean",
    "value_exp_placement_mean",
)

DISTRIBUTION_METRICS = FINGERPRINT_METRICS + (
    "placement",
    "survival_frac",
    "kills",
    "mean_army_share_alive",
    "mean_land_share_alive",
    "frac_ticks_army_rank1",
    "game_length",
)


def _rows_of(rows: list[dict], group: str) -> list[dict]:
    return [r for r in rows if r["group"] == group]


def _mean_se(rows: list[dict], key: str, seed: int = 0) -> dict:
    est, se = stats.bootstrap_se_by_game(
        [(r["game_id"], float(r[key])) for r in rows], seed=seed
    )
    return {"mean": est, "se": se, "n": len(rows)}


def headline(player_rows: list[dict], group_labels: list[str]) -> dict:
    n_games = len({r["game_id"] for r in player_rows})
    n_draws = len({r["game_id"] for r in player_rows if r["is_draw_game"]})
    per_group = {}
    for g in group_labels:
        rows = _rows_of(player_rows, g)
        wins = sum(r["won"] for r in rows)
        per_group[g] = {
            "player_games": len(rows),
            "wins": wins,
            "win_rate": wins / n_games,
            "placement": _mean_se(rows, "placement"),
            "survival_frac": _mean_se(rows, "survival_frac"),
            "kills_per_player_game": _mean_se(rows, "kills"),
        }
    return {"n_games": n_games, "n_draws": n_draws, "groups": per_group}


def paired_by_map(player_rows: list[dict], group_labels: list[str]) -> dict | None:
    """Two-group paired comparison across map pairs (same replay_id)."""
    if len(group_labels) != 2:
        return None
    a, b = group_labels

    winner_by_game: dict[str, str | None] = {}
    games_by_map: dict[str, set[str]] = defaultdict(set)
    placement_rows: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in player_rows:
        games_by_map[r["replay_id"]].add(r["game_id"])
        if r["won"]:
            winner_by_game[r["game_id"]] = r["group"]
        elif r["is_draw_game"]:
            winner_by_game.setdefault(r["game_id"], None)
        placement_rows[(r["replay_id"], r["group"])].append(float(r["placement"]))

    sweeps = Counter()
    for gids in games_by_map.values():
        winners = {winner_by_game.get(gid) for gid in gids}
        if winners == {a}:
            sweeps[a] += 1
        elif winners == {b}:
            sweeps[b] += 1
        else:
            sweeps["split"] += 1

    deltas = [
        (sum(placement_rows[(rid, a)]) / len(placement_rows[(rid, a)]))
        - (sum(placement_rows[(rid, b)]) / len(placement_rows[(rid, b)]))
        for rid in games_by_map
    ]
    return {
        "n_maps": len(games_by_map),
        "sweeps": {a: sweeps[a], b: sweeps[b], "split": sweeps["split"]},
        "sweep_sign_test_p": stats.sign_test(sweeps[a], sweeps[b]),
        "placement_delta": stats.bootstrap_paired_diff(deltas),  # a − b; negative favors a
    }


def conversion(stage_rows: list[dict], group_labels: list[str]) -> dict:
    """P(eventual win | unique mid-span leader at k alive), pooled and per group.

    Draw games are excluded (no winner exists to convert to).
    """
    out: dict = {}
    rows = [
        r for r in stage_rows if r["axis"] == "alive" and not r["is_draw_game"]
    ]
    stages = sorted({r["stage"] for r in rows}, reverse=True)
    for key in ECONOMY_KEYS:
        per_stage = []
        for k in stages:
            leaders = [r for r in rows if r["stage"] == k and r[f"{key}_leader_mid"]]
            entry = {
                "stage": k,
                "pooled": {
                    "n": len(leaders),
                    "won": sum(r["eventual_win"] for r in leaders),
                },
            }
            for g in group_labels:
                lg = [r for r in leaders if r["group"] == g]
                entry[g] = {"n": len(lg), "won": sum(r["eventual_win"] for r in lg)}
            per_stage.append(entry)
        out[key] = per_stage
    return out


def kills(player_rows: list[dict], group_labels: list[str]) -> dict:
    # killer→victim matrix needs the killer's group: join seats within a game
    group_of = {(r["game_id"], r["seat"]): r["group"] for r in player_rows}
    matrix = {ga: {gb: 0 for gb in group_labels} for ga in group_labels}
    for r in player_rows:
        ks = r["killed_by_seat"]
        if ks >= 0:
            matrix[group_of[(r["game_id"], ks)]][r["group"]] += 1

    winner_kills = {g: Counter() for g in group_labels}
    for r in player_rows:
        if r["won"]:
            winner_kills[r["group"]][int(r["kills"])] += 1

    return {
        "matrix": matrix,  # matrix[killer_group][victim_group] = eliminations
        "winner_kills_hist": {g: dict(sorted(c.items())) for g, c in winner_kills.items()},
    }


def stage_shares(stage_rows: list[dict], group_labels: list[str]) -> dict:
    """Span-mean economy shares per (group, stage), mean ± bootstrap SE."""
    out: dict = {}
    stage_rows = [r for r in stage_rows if r["axis"] == "alive"]
    stages = sorted({r["stage"] for r in stage_rows}, reverse=True)
    for key in ECONOMY_KEYS:
        per_stage = []
        for k in stages:
            entry: dict = {"stage": k}
            for g in group_labels:
                rows = [r for r in stage_rows if r["stage"] == k and r["group"] == g]
                est, se = stats.bootstrap_se_by_game(
                    [(r["game_id"], r[f"{key}_share_span"]) for r in rows]
                )
                entry[g] = {"mean": est, "se": se, "n": len(rows)}
            per_stage.append(entry)
        out[key] = per_stage
    return out


def fingerprints(player_rows: list[dict], group_labels: list[str]) -> dict:
    out = {}
    for metric in FINGERPRINT_METRICS:
        out[metric] = {
            g: stats.median_iqr([float(r[metric]) for r in _rows_of(player_rows, g)])
            for g in group_labels
        }
    return out


def outliers(player_rows: list[dict], pass_rate_min: float = 0.6, cap: int = 15) -> list[dict]:
    """High-pass-rate player-games (the 'always-passing bot' scan)."""
    flagged = [
        r for r in player_rows
        if not math.isnan(r["pass_rate"]) and r["pass_rate"] >= pass_rate_min
    ]
    flagged.sort(key=lambda r: -r["pass_rate"])
    return [
        {
            "game_id": r["game_id"],
            "seat": r["seat"],
            "group": r["group"],
            "pass_rate": round(r["pass_rate"], 3),
            "placement": r["placement"],
            "survival_frac": round(r["survival_frac"], 3),
        }
        for r in flagged[:cap]
    ]


def seat_appendix(player_rows: list[dict]) -> list[dict]:
    """Per policy slot (the run.log 'bc-model N' identity): the noise yardstick."""
    out = []
    for idx in sorted({r["policy_idx"] for r in player_rows}):
        rows = [r for r in player_rows if r["policy_idx"] == idx]
        out.append(
            {
                "policy_idx": idx,
                "group": rows[0]["group"],
                "wins": sum(r["won"] for r in rows),
                "mean_placement": sum(r["placement"] for r in rows) / len(rows),
            }
        )
    return out


def distributions(player_rows: list[dict], group_labels: list[str]) -> dict:
    out = {}
    for metric in DISTRIBUTION_METRICS:
        per_group = {}
        for g in group_labels:
            rows = _rows_of(player_rows, g)
            summ = stats.summarize([float(r[metric]) for r in rows])
            if summ["n"]:
                by_val = sorted(
                    (r for r in rows if not math.isnan(float(r[metric]))),
                    key=lambda r: float(r[metric]),
                )
                summ["min_game"] = by_val[0]["game_id"]
                summ["max_game"] = by_val[-1]["game_id"]
            per_group[g] = summ
        out[metric] = per_group
    return out


def aggregate_all(
    player_rows: list[dict], stage_rows: list[dict], group_labels: list[str]
) -> dict:
    return {
        "headline": headline(player_rows, group_labels),
        "paired_by_map": paired_by_map(player_rows, group_labels),
        "conversion": conversion(stage_rows, group_labels),
        "kills": kills(player_rows, group_labels),
        "stage_shares": stage_shares(stage_rows, group_labels),
        "fingerprints": fingerprints(player_rows, group_labels),
        "outliers": outliers(player_rows),
        "seat_appendix": seat_appendix(player_rows),
        "distributions": distributions(player_rows, group_labels),
    }
