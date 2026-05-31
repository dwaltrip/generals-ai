"""Batched eval runner — drive many games concurrently to fill the GPU.

`run_batched` keeps a fixed-size pool of concurrent games, advancing every live
game one tick per iteration. Each tick it gathers the alive NN perspectives
across all games, groups them by model so each distinct model runs a single
batched forward + decode, scatters the decoded rows back, and steps every game.
CPU policies (e.g. an eval bot) are driven per tick via `act`. Finished games
are yielded as they complete — out of order — and their pool slot is refilled
from the pending queue, keeping the batch full through the ragged tail.

Timesteps are not synchronized across slots: a freshly refilled game at t=0
runs alongside one mid-game. Each game owns its own State and perspectives, and
the forward is row-independent, so mixing them in one batch is identical to
running each alone.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TypeGuard

from game_types import ObsBundle, StaticMap
import numpy as np

from game_runner.brain import BatchablePolicy
from game_runner.policy import GameResult, Policy
from game_runner.runner import OnTickCallback, build_result
from game_runner.sim_adapter import state_to_view
from game_runner.tick_timing import TickTiming, timing_enabled
import sim_core


@dataclass
class PendingGame:
    """One game queued for the pool. `policies` are fresh (the runner calls
    `init_for_game` when the game enters a slot). `on_tick` fires after each
    step (e.g. a MetricsCollector); `ctx` is opaque caller payload handed back
    on the matching FinishedGame.
    """

    game_id: int
    map_data: StaticMap
    policies: Sequence[Policy]
    on_tick: OnTickCallback | None = None
    ctx: Any = None


@dataclass
class FinishedGame:
    game_id: int
    result: GameResult
    policies: Sequence[Policy]
    ctx: Any


@dataclass
class _LiveGame:
    spec: PendingGame
    state: sim_core.State


@dataclass
class _Row:
    """One (game, player) NN cell gathered for the batched forward."""

    slot: int           # pool slot index
    player: int         # player index within the game
    policy: BatchablePolicy
    bundle: ObsBundle


def _is_batchable(policy: Policy) -> TypeGuard[BatchablePolicy]:
    """An NN policy exposes build_obs; a plain CPU policy (e.g. evalbot) doesn't."""
    return hasattr(policy, "build_obs")


def run_batched(
    pending: Iterable[PendingGame], *, pool_size: int, max_turns: int = 2000,
) -> Iterator[FinishedGame]:
    """Run `pending` games through a pool of `pool_size` concurrent slots,
    yielding each game as it finishes (in completion order)."""
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")

    timing = TickTiming(enabled=timing_enabled())
    pending_iter = iter(pending)
    live: list[_LiveGame | None] = [None] * pool_size

    def _fill(slot: int) -> None:
        spec = next(pending_iter, None)
        if spec is None:
            live[slot] = None
            return
        state = sim_core.new_state(spec.map_data)
        if len(spec.policies) != state.num_players:
            raise ValueError(
                f"game {spec.game_id}: expected {state.num_players} policies, "
                f"got {len(spec.policies)}"
            )
        for policy in spec.policies:
            policy.init_for_game(state, spec.map_data)
        live[slot] = _LiveGame(spec, state)

    for slot in range(pool_size):
        _fill(slot)

    while any(g is not None for g in live):
        timing.tick()

        # 1. Gather: per game, NN cells -> rows to batch; CPU cells -> act now.
        rows: list[_Row] = []
        moves: dict[int, list[tuple[int, int, int, int]]] = {}
        for slot, g in enumerate(live):
            if g is None:
                continue
            moves[slot] = []
            for p, policy in enumerate(g.spec.policies):
                if not g.state.alive[p]:
                    continue
                if _is_batchable(policy):
                    timing.start()
                    view = state_to_view(g.state, g.spec.map_data, p)
                    timing.lap("state_to_view")
                    bundle = policy.build_obs(view)
                    timing.lap("build_obs")
                    rows.append(_Row(slot, p, policy, bundle))
                else:
                    timing.start()
                    src, dst, is50 = policy.act(g.state, g.spec.map_data)
                    timing.lap("cpu_act")
                    if src != -1:
                        moves[slot].append((p, src, dst, is50))

        # 2. Forward + decode, grouped by model so each model runs once/tick.
        timing.start()
        groups: dict[str, list[_Row]] = {}
        for row in rows:
            groups.setdefault(row.policy.model_handle.model_key, []).append(row)
        for group in groups.values():
            handle = group[0].policy.model_handle
            out = handle.forward_batch(
                np.stack([r.bundle.obs for r in group]),
                np.stack([r.bundle.valid_mask for r in group]),
            )
            decisions = handle.decode_batch(
                out,
                np.stack([r.bundle.policy_mask for r in group]),
                [r.policy.decode_config for r in group],
            )
            for row, decision in zip(group, decisions, strict=True):
                src, dst, is50 = row.policy.select_action(decision)
                if src != -1:
                    moves[row.slot].append((row.player, src, dst, is50))
        timing.lap("fwd_decode")

        # 3. Step every live game; fire on_tick; yield + refill on termination.
        for slot, g in enumerate(live):
            if g is None:
                continue
            game_moves = moves[slot]
            game_moves.sort(key=lambda m: m[0])  # slot order, matching run_game
            timing.start()
            g.state.step_tick(moves=game_moves, afks=[])
            if g.spec.on_tick is not None:
                g.spec.on_tick(g.state, game_moves, g.spec.policies)
            timing.lap("step")
            if g.state.alive_count <= 1 or g.state.timestep >= max_turns:
                yield FinishedGame(
                    game_id=g.spec.game_id,
                    result=build_result(g.state, g.spec.policies),
                    policies=g.spec.policies,
                    ctx=g.spec.ctx,
                )
                _fill(slot)

    timing.report()
