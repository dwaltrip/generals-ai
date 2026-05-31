"""Opt-in per-region timing for the batched runner.

Enabled via the BATCHED_TIMING env var. A disabled timer's start/lap/tick are
no-ops behind a single `enabled` check, so the hot loop stays clean and pays
nothing when timing is off (no perf_counter calls).

`start()` marks the current time; each `lap(name)` attributes the interval since
the last start/lap to `name` and rolls the mark forward — so `start(); lap(a);
lap(b)` charges two back-to-back regions. Per-region totals + call counts feed
`report()`. `fwd_decode` is timed as one region on purpose: on MPS the forward
dispatch is async and the GPU->CPU sync lands inside decode, so timing them
apart would misattribute the wait.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import time


def timing_enabled() -> bool:
    return os.environ.get("BATCHED_TIMING", "") not in ("", "0")


@dataclass
class TickTiming:
    enabled: bool = False
    ticks: int = 0
    _mark: float = 0.0
    _total: dict[str, float] = field(default_factory=dict)
    _count: dict[str, int] = field(default_factory=dict)

    def tick(self) -> None:
        if self.enabled:
            self.ticks += 1

    def start(self) -> None:
        if self.enabled:
            self._mark = time.perf_counter()

    def lap(self, name: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self._total[name] = self._total.get(name, 0.0) + (now - self._mark)
        self._count[name] = self._count.get(name, 0) + 1
        self._mark = now

    def report(self) -> None:
        if not self.enabled:
            return
        if self.ticks == 0:
            print("[batched timing] no ticks recorded")
            return
        tot = self._total.get
        cnt = self._count.get
        nn = max(cnt("build_obs", 0), 1)
        cpu = max(cnt("cpu_act", 0), 1)
        gather = tot("state_to_view", 0.0) + tot("build_obs", 0.0) + tot("cpu_act", 0.0)
        accounted = gather + tot("fwd_decode", 0.0) + tot("step", 0.0)

        def line(label: str, total: float, per_row_ms: float | None) -> None:
            per_tick_ms = total / self.ticks * 1000
            pct = total / accounted * 100 if accounted else 0.0
            pr = f"{per_row_ms:7.3f}ms" if per_row_ms is not None else f"{'—':>9s}"
            print(f"  {label:16s} {total:7.2f}s {per_tick_ms:8.2f}ms {pr} {pct:5.1f}%")

        nn_rows, cpu_rows = cnt("build_obs", 0), cnt("cpu_act", 0)
        print()
        print(f"[batched timing] {self.ticks} ticks  |  "
              f"NN rows {nn_rows} ({nn_rows / self.ticks:.1f}/tick), "
              f"CPU rows {cpu_rows} ({cpu_rows / self.ticks:.1f}/tick)")
        print(f"  {'region':16s} {'total':>8s} {'per-tick':>10s} {'per-row':>9s} {'%':>6s}")
        line("state_to_view", tot("state_to_view", 0.0), tot("state_to_view", 0.0) / nn * 1000)
        line("build_obs", tot("build_obs", 0.0), tot("build_obs", 0.0) / nn * 1000)
        line("cpu_act", tot("cpu_act", 0.0), tot("cpu_act", 0.0) / cpu * 1000)
        line("gather (Σ)", gather, None)
        line("fwd_decode", tot("fwd_decode", 0.0), None)
        line("step", tot("step", 0.0), None)
        line("accounted", accounted, None)
