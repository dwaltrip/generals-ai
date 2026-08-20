"""Per-player ground-truth derivers off the raw sim: total army (from the sim and,
for parity, from the encoded obs the model sees) and total owned land (tile count).
All read the same quantities the leaderboard does, so analyses built on any agree.
"""

from __future__ import annotations

import numpy as np

from training.analysis.fq.derivers import Deriver, per_game
from training.analysis.obs_utils import army_totals_ch_indices, per_player_army
from training.bc.obs_config import OBS_CONFIG_DEFAULTS


# TODO(fq-army-ch-obs-cfg): resolved at import from the DEFAULT obs layout, not
# from the obs_cfg a table is actually built with, and nothing checks that the
# two agree — an obs config with a different channel layout would silently read
# the wrong planes. (`dense_history_n` specifically doesn't shift these indices;
# see `army_totals_ch_indices`.)
ARMY_CH = army_totals_ch_indices(OBS_CONFIG_DEFAULTS.dense_history_n)


def kth_lowest_alive(values: np.ndarray, alive: np.ndarray, k: int) -> np.ndarray:
    """`[N]` channel index of the k-th smallest *alive* value per frame.

    `values`/`alive` are `[N, 8]` (values, bool alive mask); `k=0` is the lowest,
    `k=1` the 2nd-lowest, `k=2` the 3rd. Dead channels are pushed to `+inf` so they
    sort last and never win. Single-sources the `level` / `2nd_lowest` / `3rd_lowest`
    order-statistic baselines for the who-dies-next heuristics.

    Frames with `< k+1` alive channels can't have a real k-th-lowest; those rows get
    the sentinel `-1` (a value that never equals a real victim channel 0..7), so a
    caller comparing the pick against the true victim treats them as a miss.
    """
    masked = np.where(alive, values, np.inf)
    order = np.argsort(masked, axis=1, kind="stable")    # ascending; dead slots last
    pick = order[:, k]
    n_alive = alive.sum(axis=1)
    return np.where(n_alive >= k + 1, pick, -1)


def army_totals_per_tick(sim: dict[str, np.ndarray]) -> np.ndarray:
    """`[T, 8]` total army per slot per tick (sum of armies over owned cells).

    The canonical army-from-sim ground truth — mirrors the leaderboard army column:
    neutral/mountain cells excluded by the `ownership == slot` mask, owned cities
    contribute their garrison. The one source for sim army totals, so analyses agree.
    """
    own, arm = sim["ownership"], sim["armies"]  # [T, H, W]
    T = own.shape[0]
    tot = np.zeros((T, 8), dtype=np.int64)
    for s in range(8):
        tot[:, s] = np.where(own == s, arm, 0).reshape(T, -1).sum(1)
    return tot


# army from the encoded obs — what the MODEL sees (carries fp16 storage noise
# when obs_dtype is fp16). Scaled back to raw-army units to match army_sim.
ARMY_OBS = Deriver(
    "army_obs",
    per_player=True,
    fn=lambda f: per_player_army(f.obs, f.valid_mask, ARMY_CH).numpy().astype(float) * 1000.0,
)

# army from raw sim — GROUND TRUTH. Whole-game totals, indexed per frame.
ARMY_SIM = per_game(
    "army_sim",
    per_player=True,
    prepare=army_totals_per_tick,                              # sim -> [T, 8]
    index=lambda totals, f: totals[f.t, f.raw_order].astype(float),
)


def land_totals_per_tick(sim: dict[str, np.ndarray]) -> np.ndarray:
    """`[T, 8]` total owned tiles per slot per tick (count of owned cells).

    The land counterpart of `army_totals_per_tick` — the leaderboard land column.
    Same `ownership == slot` mask, but counts cells rather than summing army.
    """
    own = sim["ownership"]  # [T, H, W]
    T = own.shape[0]
    tot = np.zeros((T, 8), dtype=np.int64)
    for s in range(8):
        tot[:, s] = (own == s).reshape(T, -1).sum(1)
    return tot


# land from raw sim — GROUND TRUTH. Whole-game totals, indexed per frame.
LAND_SIM = per_game(
    "land_sim",
    per_player=True,
    prepare=land_totals_per_tick,                             # sim -> [T, 8]
    index=lambda totals, f: totals[f.t, f.raw_order].astype(float),
)


def captured_per_tick(sim: dict[str, np.ndarray]) -> np.ndarray:
    """`[T, 8]` bool: slot s has been captured by another player at/before tick t.

    The sim's `capture_events` is `[K, 3]` rows of `(tick, captor, victim)`; a
    capture is permanent (tiles transfer, the player is out), so the flag latches
    True from the capture tick onward. This is the *capture*-based status prod's
    `opp_N_captured_by` channel exposes — by construction blind to surrender, which
    fires a `DeathEvent` but no `CaptureEvent` and leaves army on the board. So a
    surrendered-present slot (`~alive & army>0`) stays False here, which is the
    whole point: `captured ⟹ army==0` (capture zeroes the total), so the channel
    cannot flag a player who left with army still standing.

    Latches at `tick + 1`, not `tick`: the board snapshot transfers the captured
    tiles one tick after the capture event (the obs/alive tick seam, 6.18-6), so
    army zeroes at `tick + 1`. Aligning the flag there keeps `captured ⟹ army==0`
    exact and matches the snapshot the obs army channel reads.
    """
    ev = sim["capture_events"]              # [K, 3] = (tick, captor, victim)
    T = sim["ownership"].shape[0]
    out = np.zeros((T, 8), dtype=bool)
    for tick, _captor, victim in ev:
        out[tick + 1 :, victim] = True
    return out


# capture-based status from raw sim — GROUND TRUTH. Prod's `opp_N_captured_by`
# channel collapsed to a binary flag (captor identity dropped; §6). Blind to
# surrender by construction — see `captured_per_tick`.
CAPTURED = per_game(
    "captured",
    per_player=True,
    prepare=captured_per_tick,                               # sim -> [T, 8] bool
    index=lambda flags, f: flags[f.t, f.raw_order],
)
