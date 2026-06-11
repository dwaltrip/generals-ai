"""Render aggregated eval-run analysis as markdown + PNGs.

report.md carries the headline tables; distributions.md the full percentile
appendix. Plots are kept to the genuinely curve-shaped findings (early land
share over ticks, army share over the alive-count stages).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def _f(x: float | None, nd: int = 3) -> str:
    if x is None or x != x:  # nan
        return "—"
    return f"{x:.{nd}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _conv_cell(d: dict) -> str:
    return f"{d['won'] / d['n']:.2f} ({d['won']}/{d['n']})" if d["n"] else "—"


def render_report(agg: dict, config: dict, groups: list, run_name: str) -> str:
    labels = [g.label for g in groups]
    hl = agg["headline"]
    out: list[str] = [f"# Eval analysis — {run_name}\n"]

    out.append(
        f"{hl['n_games']} games · {len(config['policy_specs'])} players · "
        f"maps `{config.get('maps', config.get('maps_arg', '?'))}` · "  # maps_arg: key in older run dirs
        f"max_turns {config.get('max_turns', '?')} · "
        f"{hl['n_draws']} draws\n"
    )
    out.append(_table(
        ["group", "policy spec", "slots"],
        [[f"**{g.label}**", f"`{g.spec}`", ", ".join(map(str, g.policy_indices))] for g in groups],
    ))

    out.append("\n## Headline\n")
    def _mean_se(d: dict) -> str:
        return f"{_f(d['mean'], 2)} ± {_f(d['se'], 2)}"

    rows = []
    for g in labels:
        d = hl["groups"][g]
        rows.append([
            g,
            f"{d['wins']} ({d['win_rate']:.1%})",
            _mean_se(d["placement"]),
            _mean_se(d["survival_frac"]),
            _mean_se(d["kills_per_player_game"]),
        ])
    out.append(_table(
        ["group", "wins", "mean placement (1=best)", "mean survival frac", "kills / player-game"],
        rows,
    ))

    paired = agg["paired_by_map"]
    if paired:
        a, b = labels[:2]
        sw = paired["sweeps"]
        pd = paired["placement_delta"]
        out.append(
            f"\n**Paired by map** ({paired['n_maps']} maps): "
            f"{a} sweeps {sw[a]}, {b} sweeps {sw[b]}, splits {sw['split']} → "
            f"sign test p = {paired['sweep_sign_test_p']:.2f}.\n\n"
            f"Per-map Δ mean placement ({a} − {b}; negative favors {a}): "
            f"{_f(pd.get('mean'), 3)} ± {_f(pd.get('se'), 3)} "
            f"(95% CI [{_f(pd['ci95'][0], 3)}, {_f(pd['ci95'][1], 3)}])\n"
        )

    out.append("\n## Conversion: does a mid-span lead become a win?\n")
    out.append(
        "P(eventual win | unique leader at the mid-span snapshot of the k-alive stage). "
        "Draw games excluded.\n"
    )
    for key, per_stage in agg["conversion"].items():
        rows = [
            [str(e["stage"]), _conv_cell(e["pooled"])] + [_conv_cell(e[g]) for g in labels]
            for e in per_stage
            if e["stage"] >= 2
        ]
        out.append(f"\n**{key} lead**\n\n" + _table(["k alive", "pooled"] + labels, rows))

    out.append("\n## Kills\n")
    km = agg["kills"]["matrix"]
    out.append("Eliminations, killer group → victim group:\n\n" + _table(
        ["killer \\ victim"] + labels,
        [[f"**{ga}**"] + [str(km[ga][gb]) for gb in labels] for ga in labels],
    ))
    out.append("\nWinner's elimination count (of the 5 in a won game):\n\n")
    hist = agg["kills"]["winner_kills_hist"]
    ks = sorted({k for h in hist.values() for k in h})
    out.append(_table(
        ["group"] + [f"{k} kills" for k in ks],
        [[g] + [str(hist[g].get(k, 0)) for k in ks] for g in labels],
    ))

    out.append("\n## Economy share by stage\n")
    out.append(
        "Span-mean share of the player-owned total, per surviving player "
        "(uniform share = 1/k). See `army_share_by_stage.png`, `early_land_share.png`.\n"
    )
    for key, per_stage in agg["stage_shares"].items():
        rows = []
        for e in per_stage:
            if e["stage"] < 2:
                continue
            rows.append(
                [str(e["stage"])]
                + [f"{_f(e[g]['mean'])} ± {_f(e[g]['se'])}" for g in labels]
                + [str(e[labels[0]]["n"])]
            )
        out.append(f"\n**{key}**\n\n" + _table(["k alive"] + labels + [f"n ({labels[0]})"], rows))

    out.append("\n## Behavioral fingerprints (median [IQR] per player-game)\n")
    out.append(_table(
        ["metric"] + labels,
        [[m] + [agg["fingerprints"][m][g] for g in labels] for m in agg["fingerprints"]],
    ))

    out.append("\n## Outliers — high pass-rate player-games\n")
    if agg["outliers"]:
        out.append(_table(
            ["game", "seat", "group", "pass rate", "placement", "survival frac"],
            [
                [o["game_id"], str(o["seat"]), o["group"], _f(o["pass_rate"]),
                 _f(o["placement"], 1), _f(o["survival_frac"], 2)]
                for o in agg["outliers"]
            ],
        ))
        out.append(
            "\nReplays: `games/<game>.html` (if rendered) or "
            "`view_game.py games/<game>.npz`.\n"
        )
    else:
        out.append("None above threshold.\n")

    out.append("\n## Per-slot appendix (noise yardstick)\n")
    out.append(
        "Same checkpoint occupies multiple slots; spread across slots of one "
        "group is pure noise.\n\n"
    )
    out.append(_table(
        ["slot", "group", "wins", "mean placement"],
        [
            [str(s["policy_idx"]), s["group"], str(s["wins"]), _f(s["mean_placement"], 2)]
            for s in agg["seat_appendix"]
        ],
    ))
    out.append(
        "\nFull percentile tables: `distributions.md`. Raw rows: "
        "`player_games.csv`, `stage_snapshots.csv`. Aggregates: `metrics.json`.\n"
    )
    return "\n".join(out)


def render_distributions(agg: dict, group_labels: list[str]) -> str:
    out = ["# Distribution appendix — per player-game metrics\n"]
    stat_keys = ("mean", "sd", "min", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "max")
    cols = ["group", "n", *stat_keys, "min game", "max game"]
    for metric, per_group in agg["distributions"].items():
        rows = []
        for g in group_labels:
            s = per_group[g]
            if not s["n"]:
                rows.append([g, "0"] + ["—"] * (len(cols) - 2))
                continue
            rows.append(
                [g, str(s["n"])]
                + [_f(s[k]) for k in stat_keys]
                + [s["min_game"], s["max_game"]]
            )
        out.append(f"\n## {metric}\n\n" + _table(cols, rows))
    return "\n".join(out)


def make_plots(
    stage_rows: list[dict], agg: dict, group_labels: list[str], out_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []

    # Early-game land share vs tick (tick-axis rows)
    tick_rows = [r for r in stage_rows if r["axis"] == "tick"]
    if tick_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        for g in group_labels:
            by_tick: dict[int, list[float]] = defaultdict(list)
            for r in tick_rows:
                if r["group"] == g:
                    by_tick[r["stage"]].append(r["land_share_mid"])
            ticks = sorted(by_tick)
            ax.plot(ticks, [sum(by_tick[t]) / len(by_tick[t]) for t in ticks], marker="o", label=g)
        ax.set_xlabel("tick")
        ax.set_ylabel("mean land share")
        ax.set_title("Early-game land share")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "early_land_share.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    # Army share across alive-count stages, mean ± SE
    fig, ax = plt.subplots(figsize=(7, 4))
    per_stage = agg["stage_shares"]["army"]
    stages = [e["stage"] for e in per_stage if e["stage"] >= 2]
    for g in group_labels:
        means = [e[g]["mean"] for e in per_stage if e["stage"] >= 2]
        ses = [e[g]["se"] for e in per_stage if e["stage"] >= 2]
        ax.errorbar(stages, means, yerr=ses, marker="o", capsize=3, label=g)
    ax.plot(stages, [1 / k for k in stages], linestyle=":", color="gray", label="uniform 1/k")
    ax.invert_xaxis()
    ax.set_xlabel("players alive")
    ax.set_ylabel("span-mean army share")
    ax.set_title("Army share by stage")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "army_share_by_stage.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p)
    return written
