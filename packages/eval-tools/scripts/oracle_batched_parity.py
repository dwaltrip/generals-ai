#!/usr/bin/env -S uv run python
"""Multi-game oracle: does the batched runner reproduce run_game, game-for-game?

Disposable dev gate for the batched runner. For each game spec it runs the game
through `run_batched` AND independently through `run_game`, then compares the
full per-game artifact. The chain of trust: `test_parity` pins run_game ≡
pre-refactor; this pins batched ≡ run_game.

Runs on CPU (run-to-run reproducible) with force_move + argmax (no RNG), NN-only
except the one evalbot-plumbing case (np.random seeded). Cases:

  C1  pool=1 NN-vs-NN          reduces to run_game
  C2  pool=1 NN-vs-evalbot     CPU-row passthrough (np seeded)
  C3  pool=2, 5 NN-vs-NN       refill, out-of-order completion, game_id tracking
  C4  pool=2, 3 e3-vs-e5       group-by-model_key (2 forwards/tick), scatter
  C5  pool=2, 2× 4p NN-only    many perspectives/game, wide gather/scatter

Compare: moves + winner + length + player_stats + curves exact (on divergence,
report first (game, tick) + the moves — tiny/late ⇒ benign forward-noise flip,
early/structural ⇒ scatter bug); diagnostics allclose (the ~1e-4 batched-forward
delta flows into them).

Run (from repo root):
    ./packages/eval-tools/scripts/oracle_batched_parity.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval_tools.metrics_collector import MetricsCollector
from eval_tools.policy_spec import parse_policy_spec
from game_runner.batched import PendingGame, run_batched
from game_runner.runner import run_game
from game_runner.seed_map import list_replay_ids_by_player_count, load_static_from_db


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CKPTS = _REPO_ROOT / "training/data/runs-cloud/2026-05-28T08-21-33Z/checkpoints"
_E5, _E3 = _CKPTS / "epoch_005.pt", _CKPTS / "epoch_003.pt"
_DEVICE = torch.device("cpu")
_SAMPLE_INTERVAL = 25


def _ckpt(path: Path) -> str:
    return f"checkpoint:{path}:force_move=true,value_head_variant=pyramid"


@dataclass
class _Artifact:
    moves: list[list[tuple[int, ...]]]
    result: Any
    metrics: dict


def _build_policies(specs: list[str]) -> list:
    return [parse_policy_spec(s, slot, _DEVICE) for slot, s in enumerate(specs)]


def _logging_on_tick(log: list, coll: MetricsCollector):
    def on_tick(state, moves, policies) -> None:
        log.append([tuple(int(x) for x in m) for m in moves])
        coll.on_tick(state, moves, policies)
    return on_tick


def _play_ref(replay_id: str, specs: list[str], max_turns: int, seed_np: int | None) -> _Artifact:
    static = load_static_from_db(replay_id)
    policies = _build_policies(specs)
    coll = MetricsCollector(len(specs), _SAMPLE_INTERVAL)
    log: list = []
    if seed_np is not None:
        np.random.seed(seed_np)
    result = run_game(policies, static, max_turns=max_turns, on_tick=_logging_on_tick(log, coll))
    return _Artifact(log, result, coll.finalize(result, policies))


def _play_batched(
    specs_list: list[tuple[str, list[str]]], pool_size: int, max_turns: int, seed_np: int | None,
) -> dict[int, _Artifact]:
    pendings = []
    for gid, (replay_id, specs) in enumerate(specs_list):
        static = load_static_from_db(replay_id)
        policies = _build_policies(specs)
        coll = MetricsCollector(len(specs), _SAMPLE_INTERVAL)
        log: list = []
        pendings.append(PendingGame(
            game_id=gid, map_data=static, policies=policies,
            on_tick=_logging_on_tick(log, coll), ctx=(log, coll),
        ))
    if seed_np is not None:
        np.random.seed(seed_np)
    arts: dict[int, _Artifact] = {}
    for fin in run_batched(pendings, pool_size=pool_size, max_turns=max_turns):
        log, coll = fin.ctx
        arts[fin.game_id] = _Artifact(log, fin.result, coll.finalize(fin.result, fin.policies))
    return arts


def _first_move_divergence(a: list, b: list) -> int | None:
    n = min(len(a), len(b))
    for t in range(n):
        if a[t] != b[t]:
            return t
    return None if len(a) == len(b) else n


def _diff(a: Any, b: Any, path: str) -> list[str]:
    """Recursive compare: numbers within tolerance (nan-aware), rest exact."""
    if isinstance(a, dict):
        if set(a) != set(b):
            return [f"{path}: keys {set(a) ^ set(b)}"]
        out: list[str] = []
        for k in a:
            out += _diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out += _diff(x, y, f"{path}[{i}]")
        return out
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        if math.isnan(a) and math.isnan(b):
            return []
        return [] if math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-3) else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def _compare_game(bat: _Artifact, ref: _Artifact) -> list[str]:
    fails: list[str] = []
    t = _first_move_divergence(bat.moves, ref.moves)
    if t is not None:
        bm = bat.moves[t] if t < len(bat.moves) else "END"
        rm = ref.moves[t] if t < len(ref.moves) else "END"
        lens = f"{len(bat.moves)}/{len(ref.moves)}"
        fails.append(f"moves @tick {t}: batched={bm} ref={rm} (lens {lens})")
    if bat.result.winner != ref.result.winner:
        fails.append(f"winner {bat.result.winner} != {ref.result.winner}")
    if bat.result.game_length != ref.result.game_length:
        fails.append(f"length {bat.result.game_length} != {ref.result.game_length}")
    for p, (x, y) in enumerate(zip(bat.result.player_stats, ref.result.player_stats, strict=True)):
        if (x.land, x.army) != (y.land, y.army):
            fails.append(f"p{p} stats ({x.land},{x.army}) != ({y.land},{y.army})")
    fails += _diff(bat.metrics, ref.metrics, "metrics")
    return fails


def _run_case(
    name: str, specs_list: list[tuple[str, list[str]]], pool_size: int,
    max_turns: int, seed_np: int | None = None,
) -> bool:
    print(f"\n=== {name}  ({len(specs_list)} games, pool={pool_size}, max_turns={max_turns}) ===")
    batched = _play_batched(specs_list, pool_size, max_turns, seed_np)
    ok = True
    for gid, (replay_id, specs) in enumerate(specs_list):
        ref = _play_ref(replay_id, specs, max_turns, seed_np)
        fails = _compare_game(batched[gid], ref)
        status = "PASS" if not fails else "FAIL"
        print(f"  game {gid}: {status}  len={batched[gid].result.game_length}")
        for f in fails[:6]:
            print(f"      - {f}")
        ok = ok and not fails
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="batched-runner parity oracle")
    parser.add_argument("--max-turns", type=int, default=100,
                        help="per-game cap; higher exercises ragged/out-of-order completion")
    mt = parser.parse_args().max_turns

    for ck in (_E5, _E3):
        if not ck.exists():
            print(f"checkpoint missing: {ck}")
            return 1
    two = list_replay_ids_by_player_count(2)
    four = list_replay_ids_by_player_count(4)
    if len(two) < 5 or len(four) < 2:
        print("corpus lacks enough 2p/4p maps")
        return 1
    e5, e3 = _ckpt(_E5), _ckpt(_E3)

    cases = [
        _run_case("C1 reduces-to-run_game", [(two[0], [e5, e5])], 1, mt),
        _run_case("C2 evalbot plumbing", [(two[0], [e5, "evalbot"])], 1, mt, seed_np=0),
        _run_case("C3 refill+ragged", [(r, [e5, e5]) for r in two[:5]], 2, mt),
        _run_case("C4 multi-model (e3 vs e5)", [(r, [e3, e5]) for r in two[:3]], 2, mt),
        _run_case("C5 4p multi-perspective", [(r, [e5, e5, e5, e5]) for r in four[:2]], 2, mt),
    ]
    ok = all(cases)
    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
