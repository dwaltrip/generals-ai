# sim-core

Rust + PyO3 simulator for generals.io replays. Single entrypoint:
`sim_core.simulate(replay) -> State`. Replays the wire-format move/AFK
stream and produces per-timestep snapshots, end-of-game state vectors,
event lists (captures, neutralizes, deaths), and a damage matrix.

See `src/lib.rs` for the entrypoint and `src/state.rs` for the State
struct + step body.

## Timestep ↔ snapshot semantics

The simulator records one snapshot per timestep, plus an initial snapshot
at `t=0` taken before the first step. `step()` runs:

```
process AFKs / buffer moves     ← AFK + capture events fire here
resolve moves                     (event.timestep = pre-increment value)
timestep += 1
apply production                ← uses post-increment timestep
snapshot()                      ← captures post-step state
```

So `snapshot[t]` = simulator state **at** timestep `t`, after the step
that took the sim from `t-1` to `t` completes. Because events fire
before the increment but production fires after it, where you observe
each event's effect differs by event type:

| Event type                     | `.timestep` field semantics | Effect visible in |
|--------------------------------|------------------------------|--------------------|
| Land-tick production (K%50==0) | (no event object)            | `snapshot[K]`      |
| Per-2-step production (t%2==0) | (no event object)            | `snapshot[t]`      |
| Capture event at `e`           | pre-increment value          | `snapshot[e+1]`    |
| Death event at `e`             | pre-increment value          | `snapshot[e+1]`    |
| Neutralize event at `e`        | pre-increment value          | `snapshot[e+1]`    |

`snapshot[e]` is the **pre-event** state for capture / death /
neutralize events. `replay-parser/replay_parser/invariants.py` relies on
this convention when checking general-tile flips.
