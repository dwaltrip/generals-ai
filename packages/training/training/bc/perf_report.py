"""Build `perf.md` — the per-run performance report.

Reads a run dir's structured outputs — args, host fingerprint, per-batch and
per-epoch logs, the GPU/contention sidecar, and the timing profiler summary —
and renders the perf/starvation tables we'd otherwise recompute by hand: host
draw, throughput, GPU utilization, host contention, obs-pipeline starvation,
and the producer obs-build breakdown, capped with a heuristic verdict.
Training quality lives in its own report (`analysis.quality_report`).

Structure: `load -> compute -> render -> assemble`. `compute_metrics` returns a
section-keyed dict of plain scalars (never formatted strings); the renderers
turn one run's metrics into vertical tables. Keeping compute string-free is
what lets a future multi-run comparison reuse it wholesale — a transposed
renderer over a list of these dicts, no re-plumbing. Every section self-trims
when its source artifact is absent, so non-profiled or older runs still
produce a partial report.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from statistics import fmean, median, pstdev

from training.bc.constants import H_PADDED, W_PADDED, obs_channel_count
from training.shared.timing import histo_bucket_midpoint_ns
from training.shared.timing_run import _span_from_json
from utils.format import (
    format_bytes,
    format_duration,
    format_ns,
    format_ns_exact,
    format_pct,
    md_table,
)


log = logging.getLogger(__name__)


# Verdict thresholds — tunable module constants, refined as we see more runs.
_STARVED_UTIL_PCT = 70.0        # mean GPU util below this ...
_STARVED_FETCH_FRAC = 0.25      # ... plus fetch_wait above this share of wall = starved
_GPU_BOUND_UTIL_PCT = 85.0      # mean util at/above this = healthy/GPU-bound
_BAD_DRAW_RAM_BYTES = 1024**4   # < 1 TiB RAM flags an off-pool host draw
_BAD_DRAW_STEAL_PCT = 5.0       # mean CPU steal above this = contended host
_WARMUP_BATCHES = 200           # leading batches dropped for steady-state sps
_UTIL_WARMUP_SEC = 40           # leading sidecar seconds dropped for util/steal


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #


@dataclass
class RunArtifacts:
    """A run dir's structured artifacts, each None / empty-list when absent."""

    run_dir: Path
    args: dict | None
    args_cloud: dict | None
    host: dict | None
    batches: list[dict]
    epochs: list[dict]
    gpu_util: list[dict]
    prof: dict | None
    pin: dict | None

    @classmethod
    def load(cls, run_dir: Path) -> RunArtifacts:
        return cls(
            run_dir=run_dir,
            args=_read_json(run_dir / "args.json"),
            args_cloud=_read_json(run_dir / "args_cloud.json"),
            host=_read_json(run_dir / "host_info.json"),
            batches=_read_jsonl(run_dir / "batches.jsonl"),
            epochs=_read_jsonl(run_dir / "epochs.jsonl"),
            gpu_util=_read_jsonl(run_dir / "gpu_util.jsonl"),
            prof=_read_json(run_dir / "prof" / "summary.json"),
            pin=_read_json(run_dir / "prof" / "pin.json"),
        )


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict]:
    try:
        text = path.read_text()
    except OSError:
        return []
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a stray non-record line (e.g. the sidecar's util_error)
    return out


# --------------------------------------------------------------------------- #
# Compute — returns plain scalars keyed by section (compare-ready)
# --------------------------------------------------------------------------- #


def compute_metrics(a: RunArtifacts) -> dict:
    """One run's metrics as a section-keyed dict of scalars (no strings)."""
    metrics = {
        "header": _compute_header(a),
        "host": _compute_host(a),
        "throughput": _compute_throughput(a),
        "gpu_util": _compute_gpu_util(a),
        "contention": _compute_contention(a),
        "consumer": _compute_consumer(a),
        "pin": _compute_pin(a),
        "producer": _compute_producer(a),
        "dist_tails": _compute_dist_tails(a),
    }
    metrics["verdict"] = _compute_verdict(metrics)
    return metrics


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, `q` in [0, 1]."""
    s = sorted(values)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def _sps_series(batches: list[dict]) -> list[float]:
    """Per-batch samples/sec from cumulative `wall_time_sec` deltas."""
    series = []
    for prev, cur in zip(batches, batches[1:], strict=False):
        dt = cur["wall_time_sec"] - prev["wall_time_sec"]
        if dt > 0:
            series.append(cur["batch_size"] / dt)
    return series


def _compute_header(a: RunArtifacts) -> dict:
    args = a.args or {}
    obs = args.get("arch", {}).get("obs", {}) or {}
    n = obs.get("dense_history_n")
    return {
        "run_id": a.run_dir.name,
        "dense_history_n": n,
        "obs_channels": obs_channel_count(n) if n is not None else None,
        "obs_dtype": obs.get("obs_dtype"),
        "batch_size": args.get("batch_size"),
        "num_workers": args.get("num_workers"),
        "precision": args.get("precision"),
        "gpu": (a.args_cloud or {}).get("gpu") or args.get("gpu"),
        "profiled": a.prof is not None,
    }


def _compute_host(a: RunArtifacts) -> dict | None:
    cloud = a.args_cloud or {}
    host = a.host or {}
    out = {
        "region": cloud.get("modal_region"),
        "gpu": cloud.get("gpu") or host.get("gpu_name"),
        "cpu_count": host.get("cpu_count"),
        "cpu_model": host.get("cpu_model"),
        "numa_nodes": host.get("numa_nodes"),
        "ram_total_bytes": host.get("ram_total_bytes"),
        "shm_total_bytes": host.get("shm_total_bytes"),
        "pcie_gen": host.get("gpu_pcie_gen"),
        "pcie_width": host.get("gpu_pcie_width"),
    }
    return out if any(v is not None for v in out.values()) else None


def _compute_throughput(a: RunArtifacts) -> dict | None:
    out: dict = {}
    if a.epochs:
        last = a.epochs[-1]
        out["overall_sps"] = last.get("samples_per_sec")
        out["mfu"] = last.get("mfu")
        out["duration_sec"] = last.get("duration_sec")
    series = _sps_series(a.batches)
    if series:
        steady = series[_WARMUP_BATCHES:] or series
        out["steady_mean"] = fmean(steady)
        out["steady_median"] = median(steady)
        out["steady_std"] = pstdev(steady) if len(steady) > 1 else 0.0
        out["steady_min"] = min(steady)
        out["steady_max"] = max(steady)
        if out.get("overall_sps") is None:  # no epochs.jsonl — derive from batches
            total_samples = sum(b.get("batch_size", 0) for b in a.batches)
            total_wall = a.batches[-1].get("wall_time_sec")
            if total_wall:
                out["overall_sps"] = total_samples / total_wall
    return out or None


def _compute_gpu_util(a: RunArtifacts) -> dict | None:
    recs = [
        r for r in a.gpu_util
        if r.get("gpu_util_pct") is not None and r.get("t_sec", 0) > _UTIL_WARMUP_SEC
    ]
    if not recs:
        return None
    utils = [r["gpu_util_pct"] for r in recs]
    mem = [r["mem_reserved_mb"] for r in recs if r.get("mem_reserved_mb") is not None]
    return {
        "mean": fmean(utils),
        "median": median(utils),
        "std": pstdev(utils) if len(utils) > 1 else 0.0,
        "p10": _percentile(utils, 0.1),
        "p90": _percentile(utils, 0.9),
        "frac_below_80": sum(u < 80 for u in utils) / len(utils),
        "mem_reserved_mb": max(mem) if mem else None,
    }


def _compute_contention(a: RunArtifacts) -> dict | None:
    steal = [r["cpu_steal_pct"] for r in a.gpu_util if r.get("cpu_steal_pct") is not None]
    load = [r["load_avg_1m"] for r in a.gpu_util if r.get("load_avg_1m") is not None]
    if not steal and not load:
        return None
    out: dict = {}
    if steal:
        out["steal_mean"] = fmean(steal)
        out["steal_max"] = max(steal)
    if load:
        out["load_mean"] = fmean(load)
        out["load_max"] = max(load)
    return out


def _compute_consumer(a: RunArtifacts) -> dict | None:
    if not a.prof or not a.prof.get("consumer"):
        return None
    cons = a.prof["consumer"]
    out: dict = {}
    s = _span_from_json(cons.get("fetch_wait", {}))
    if s.calls:
        out["fetch_wait_ms_batch"] = s.total_ns / s.calls / 1e6
        fw_total_s = s.total_ns / 1e9
        duration = a.epochs[-1].get("duration_sec") if a.epochs else None
        if duration:
            out["fetch_wait_frac"] = fw_total_s / duration
    s2 = _span_from_json(cons.get("h2d", {}))
    if s2.calls:
        out["h2d_ms_batch"] = s2.total_ns / s2.calls / 1e6
    return out or None


def _compute_pin(a: RunArtifacts) -> dict | None:
    """Pin-memory thread per-batch breakdown (`bc.pin_instrument`, written to
    `prof/pin.json` on profiled runs). Each span is ms per batch on the batch
    count (`pin_put` calls — one put per forwarded batch). `work` = copy +
    destruct is the thread's own serial cost; `work / period` near 1.0 says the
    pin thread is the throughput ceiling. `get` folds in startup/idle polling,
    so it's a coarse 'blocked on workers' signal, not a clean per-batch cost."""
    if not a.pin:
        return None
    spans = {k: _span_from_json(v) for k, v in a.pin.items()}
    put = spans.get("pin_put")
    n_batches = put.calls if put and put.calls else None
    if not n_batches:  # put absent (unexpected) — fall back to the copy count
        copy = spans.get("pin_copy")
        n_batches = copy.calls if copy and copy.calls else None
    if not n_batches:
        return None

    def ms_batch(name: str) -> float | None:
        s = spans.get(name)
        return s.total_ns / n_batches / 1e6 if s else None

    out: dict = {
        "get_ms": ms_batch("pin_get"),
        "copy_ms": ms_batch("pin_copy"),
        "destruct_ms": ms_batch("pin_destruct"),
        "put_ms": ms_batch("pin_put"),
    }
    out["work_ms"] = (out["copy_ms"] or 0.0) + (out["destruct_ms"] or 0.0)

    # Batch period from throughput — the budget the pin thread must beat to keep
    # the GPU fed. work / period near 1.0 ⇒ the pin thread is the bottleneck.
    bs = (a.args or {}).get("batch_size")
    sps = a.epochs[-1].get("samples_per_sec") if a.epochs else None
    if bs and sps:
        out["period_ms"] = bs / sps * 1000
        out["work_frac_period"] = out["work_ms"] / out["period_ms"]
    return out


# Byte-moving seams: annotate with effective GiB/s (obs bytes / time) so a slow
# byte path stands out.
_BYTE_SEAMS = frozenset({"assemble", "collate", "shm_copy"})


def _compute_producer(a: RunArtifacts) -> dict | None:
    """Producer seams split into the grouped table (leaves + TOTAL) and the
    reference spans (`grouped=False`, which overlap the grouped rows). Byte-
    moving seams carry an effective `gib_s` from the obs tensor size."""
    if not a.prof or not a.prof.get("producer"):
        return None
    prod = a.prof["producer"]
    n = a.prof.get("n_samples") or 1
    spans = {k: _span_from_json(v) for k, v in prod.items()}
    grouped = {k: s for k, s in spans.items() if s.grouped}
    ungrouped = {k: s for k, s in spans.items() if not s.grouped}
    total_ns = sum(s.total_ns for s in grouped.values())

    # Obs bytes/sample (fp32) — dominant term for the byte-moving seams (the
    # masks/scalars are <3%). None when we can't size the obs tensor.
    args = a.args or {}
    n_dense = (args.get("arch", {}).get("obs", {}) or {}).get("dense_history_n")
    obs_ch = obs_channel_count(n_dense) if n_dense is not None else None
    obs_bytes = obs_ch * H_PADDED * W_PADDED * 4 if obs_ch else None

    def gib_s(name: str, us_per_sample: float) -> float | None:
        if name not in _BYTE_SEAMS or not obs_bytes or us_per_sample <= 0:
            return None
        return obs_bytes / (us_per_sample * 1e-6) / 1024**3

    grouped_rows = []
    for name, s in sorted(grouped.items(), key=lambda kv: -kv[1].total_ns):
        us = s.total_ns / n / 1e3
        grouped_rows.append({
            "region": name, "us_per_sample": us,
            "share": s.total_ns / total_ns if total_ns else 0.0,
            "gib_s": gib_s(name, us),
        })
    grouped_rows.append({
        "region": "TOTAL", "us_per_sample": total_ns / n / 1e3,
        "share": 1.0, "gib_s": None,
    })

    reference_rows = [
        {"region": name, "us_per_sample": s.total_ns / n / 1e3}
        for name, s in sorted(ungrouped.items(), key=lambda kv: -kv[1].total_ns)
    ]
    return {"grouped": grouped_rows, "reference": reference_rows}


def _histo_percentile_ns(histo: dict[int, int], q: float) -> float | None:
    """Approximate percentile from a log2 histogram. Returns the geometric
    midpoint of the bucket containing the target rank, or None if empty."""
    total = sum(histo.values())
    if not total:
        return None
    target = q * (total - 1)
    cumulative = 0
    for bucket in sorted(histo):
        cumulative += histo[bucket]
        if cumulative > target:
            return histo_bucket_midpoint_ns(bucket)
    return histo_bucket_midpoint_ns(max(histo))


def _compute_dist_tails(a: RunArtifacts) -> list[dict] | None:
    """Per-span distribution tails (p50, p95, max) from histogram data.
    Spans from producer, consumer, and the pin thread are included — the pin
    spans' `max` is the freeze detector (e.g. a multi-second `pin_destruct`).
    When a span name appears in more than one section (num_workers=0 shares a
    single process), the first-seen entry wins to avoid duplicate rows."""
    sources: list[tuple[str, dict]] = []
    if a.prof:
        for section in ("producer", "consumer"):
            raw = a.prof.get(section)
            if raw:
                sources.append((section, raw))
    if a.pin:
        sources.append(("pin", a.pin))
    if not sources:
        return None
    seen: set[str] = set()
    rows: list[dict] = []
    for section, raw in sources:
        for name, v in raw.items():
            if name in seen:
                continue
            seen.add(name)
            s = _span_from_json(v)
            if not s.histo:
                log.warning(
                    "span %r in %s has no histogram data (old format?); "
                    "skipping distribution tails", name, section,
                )
                continue
            rows.append({
                "region": name,
                "section": section,
                "p50_ns": _histo_percentile_ns(s.histo, 0.5),
                "p95_ns": _histo_percentile_ns(s.histo, 0.95),
                "max_ns": s.max_ns,
            })
    return rows or None


def _compute_verdict(metrics: dict) -> list[str] | None:
    flags: list[str] = []
    util = metrics.get("gpu_util")
    cons = metrics.get("consumer") or {}
    if util is not None:
        frac = cons.get("fetch_wait_frac")
        if util["mean"] < _STARVED_UTIL_PCT and frac is not None and frac > _STARVED_FETCH_FRAC:
            flags.append(
                f"⚠️ PRODUCER-STARVED — GPU util {util['mean']:.0f}%, "
                f"fetch_wait {format_pct(frac)} of wall"
            )
        elif util["mean"] >= _GPU_BOUND_UTIL_PCT:
            flags.append(f"✅ GPU-BOUND — util {util['mean']:.0f}%")

    host = metrics.get("host") or {}
    cont = metrics.get("contention") or {}
    bad: list[str] = []
    ram = host.get("ram_total_bytes")
    if ram is not None and ram < _BAD_DRAW_RAM_BYTES:
        bad.append(f"RAM {format_bytes(ram)} < 1 TiB")
    if cont.get("steal_mean", 0.0) > _BAD_DRAW_STEAL_PCT:
        bad.append(f"CPU steal {cont['steal_mean']:.1f}%")
    if bad:
        region = host.get("region")
        suffix = f" (region {region})" if region else ""
        flags.append("🎲 BAD DRAW — " + ", ".join(bad) + suffix)

    return flags or None


# --------------------------------------------------------------------------- #
# Render — one run's metrics dict -> markdown blocks (None = skip section)
# --------------------------------------------------------------------------- #


def _num(x: float | None, digits: int = 0) -> str:
    """A number at `digits` precision, or an em-dash for None."""
    return f"{x:.{digits}f}" if x is not None else "—"


def _render_header(h: dict | None) -> str | None:
    if not h:
        return None
    parts: list[str] = []
    if h.get("dense_history_n") is not None:
        parts.append(f"n={h['dense_history_n']}")
    if h.get("obs_channels"):
        parts.append(f"{h['obs_channels']} ch")
    if h.get("obs_dtype"):
        parts.append(str(h["obs_dtype"]))
    if h.get("batch_size"):
        parts.append(f"bs={h['batch_size']}")
    if h.get("num_workers") is not None:
        parts.append(f"{h['num_workers']} workers")
    if h.get("precision"):
        parts.append(str(h["precision"]))
    if h.get("gpu"):
        parts.append(str(h["gpu"]))
    parts.append("profiled ✓" if h.get("profiled") else "not profiled")
    return "**config** " + " · ".join(parts)


def _render_host(h: dict | None) -> str | None:
    if not h:
        return None
    pcie = f"gen{h['pcie_gen']} x{h['pcie_width']}" if h.get("pcie_gen") else "—"
    ram = format_bytes(h["ram_total_bytes"]) if h.get("ram_total_bytes") else "—"
    shm = format_bytes(h["shm_total_bytes"]) if h.get("shm_total_bytes") else "—"
    rows = [
        ["region", h.get("region") or "—"],
        ["GPU", h.get("gpu") or "—"],
        ["CPU cores", h.get("cpu_count") if h.get("cpu_count") is not None else "—"],
        ["CPU model", h.get("cpu_model") or "—"],
        ["NUMA nodes", h.get("numa_nodes") if h.get("numa_nodes") is not None else "—"],
        ["system RAM", ram],
        ["PCIe", pcie],
        ["/dev/shm", shm],
    ]
    return "## Host draw\n\n" + md_table(["field", "value"], rows)


def _render_throughput(t: dict | None) -> str | None:
    if not t:
        return None
    rows = []
    if t.get("overall_sps") is not None:
        rows.append(["overall sps", _num(t["overall_sps"])])
    if "steady_mean" in t:
        rows.append(["steady sps (mean)", _num(t["steady_mean"])])
        rows.append(["steady sps (median)", _num(t["steady_median"])])
        rows.append(["steady sps (std)", _num(t["steady_std"])])
        rows.append(["min / max", f"{t['steady_min']:.0f} / {t['steady_max']:.0f}"])
    if t.get("duration_sec") is not None:
        rows.append(["epoch time", format_duration(t["duration_sec"])])
    if t.get("mfu") is not None:
        rows.append(["MFU", format_pct(t["mfu"])])
    return "## Throughput\n\n" + md_table(["metric", "value"], rows, align=("left", "right"))


def _render_gpu_util(u: dict | None) -> str | None:
    if not u:
        return None
    rows = [
        ["mean / median", f"{u['mean']:.0f} / {u['median']:.0f}"],
        ["std", _num(u["std"])],
        ["p10 / p90", f"{u['p10']:.0f} / {u['p90']:.0f}"],
        ["% time <80%", format_pct(u["frac_below_80"])],
    ]
    if u.get("mem_reserved_mb") is not None:
        rows.append(["mem_reserved", format_bytes(u["mem_reserved_mb"] * 1024**2)])
    return "## GPU utilization\n\n" + md_table(["metric", "value"], rows, align=("left", "right"))


def _render_contention(c: dict | None) -> str | None:
    if not c:
        return None
    rows = []
    if "steal_mean" in c:
        rows.append(["CPU steal mean / max", f"{c['steal_mean']:.1f}% / {c['steal_max']:.1f}%"])
    if "load_mean" in c:
        rows.append(["load avg mean / max", f"{c['load_mean']:.1f} / {c['load_max']:.1f}"])
    return "## Contention\n\n" + md_table(["metric", "value"], rows, align=("left", "right"))


def _render_consumer(c: dict | None) -> str | None:
    if not c:
        return None
    rows = []
    if "fetch_wait_ms_batch" in c:
        rows.append(["fetch_wait / batch", f"{c['fetch_wait_ms_batch']:.1f} ms"])
    if "fetch_wait_frac" in c:
        rows.append(["fetch_wait % wall", format_pct(c["fetch_wait_frac"])])
    if "h2d_ms_batch" in c:
        rows.append(["h2d / batch", f"{c['h2d_ms_batch']:.2f} ms"])
    return "## Starvation (consumer)\n\n" + md_table(
        ["metric", "value"], rows, align=("left", "right")
    )


def _render_pin(p: dict | None) -> str | None:
    if not p:
        return None

    def ms(x: float | None) -> str:
        return f"{x:.1f} ms" if x is not None else "—"

    rows = [
        ["get (wait for workers)", ms(p.get("get_ms"))],
        ["copy (pin_memory)", ms(p.get("copy_ms"))],
        ["destruct (free shm)", ms(p.get("destruct_ms"))],
        ["put (wait for consumer)", ms(p.get("put_ms"))],
        ["work (copy + destruct)", ms(p.get("work_ms"))],
    ]
    if p.get("period_ms") is not None:
        rows.append(["batch period", ms(p["period_ms"])])
    if p.get("work_frac_period") is not None:
        rows.append(["work / period", format_pct(p["work_frac_period"])])
    out = "## Pin-memory thread (ms/batch)\n\n" + md_table(
        ["span", "value"], rows, align=("left", "right")
    )
    out += (
        "\n\n*get/put are queue waits (blocked on workers / on the consumer); "
        "copy+destruct is the thread's own serial work. work / period near 100% "
        "⇒ the pin thread is the throughput ceiling.*"
    )
    return out


def _render_producer(p: dict | None) -> str | None:
    if not p or not p.get("grouped"):
        return None
    grouped = [
        [r["region"], f"{r['us_per_sample']:.1f}", format_pct(r["share"]),
         f"{r['gib_s']:.2f}" if r.get("gib_s") is not None else "—"]
        for r in p["grouped"]
    ]
    out = "## Producer obs-build (µs/sample)\n\n" + md_table(
        ["region", "µs/sample", "share", "GiB/s"],
        grouped, align=("left", "right", "right", "right"),
    )
    if p.get("reference"):
        ref = [[r["region"], f"{r['us_per_sample']:.1f}"] for r in p["reference"]]
        out += "\n\n### Producer — reference spans (overlap the grouped seams; not in TOTAL)\n\n"
        out += md_table(["region", "µs/sample"], ref, align=("left", "right"))
    return out


def _render_dist_tails(rows: list[dict] | None) -> str | None:
    if not rows:
        return None
    table = [
        [
            r["region"],
            format_ns(r.get("p50_ns")),
            format_ns(r.get("p95_ns")),
            format_ns_exact(r.get("max_ns")),
        ]
        for r in rows
    ]
    out = "## Distribution tails\n\n" + md_table(
        ["region", "p50", "p95", "max"],
        table, align=("left", "right", "right", "right"),
    )
    out += "\n\n*p50/p95 estimated from log2 histogram (±2× resolution); max is exact.*"
    return out


def _render_verdict(flags: list[str] | None) -> str | None:
    if not flags:
        return None
    return "## Verdict\n\n" + "\n".join(f"- {f}" for f in flags)


# --------------------------------------------------------------------------- #
# Assemble
# --------------------------------------------------------------------------- #


# (section key, renderer). Edit/reorder here to change the report layout.
_SECTIONS = [
    ("host", _render_host),
    ("throughput", _render_throughput),
    ("gpu_util", _render_gpu_util),
    ("contention", _render_contention),
    ("consumer", _render_consumer),
    ("pin", _render_pin),
    ("producer", _render_producer),
    ("dist_tails", _render_dist_tails),
    ("verdict", _render_verdict),
]


def build_perf_report(run_dir: Path) -> str:
    """Render the full markdown perf report for one run dir."""
    metrics = compute_metrics(RunArtifacts.load(run_dir))
    parts = [f"# Perf report — {run_dir.name}"]
    header = _render_header(metrics["header"])
    if header:
        parts.append(header)
    for key, renderer in _SECTIONS:
        block = renderer(metrics[key])
        if block:
            parts.append(block)
    return "\n\n".join(parts) + "\n"
