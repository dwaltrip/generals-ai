#!/usr/bin/env -S uv run python
"""De-risk: is MPS eval run-to-run deterministic at a fixed config?

Plays fully deterministic NN-vs-NN games (force_move + argmax, so no RNG) twice
per map on MPS and reports how much the two runs diverge — move-level (first
divergent tick, fraction of ticks differing) and outcome-level (winner, length).

Why it matters: the batched runner can't be expected to reproduce the sequential
runner byte-for-byte on MPS if MPS itself isn't run-to-run deterministic. This
measures the baseline noise that's *already* in MPS eval, independent of
batching. A CPU control pair confirms the methodology (CPU should be identical).

  - identical MPS runs  -> MPS is effectively deterministic for eval; batching
    adds no new nondeterminism in kind.
  - divergent MPS runs  -> MPS eval carries inherent run-to-run noise, so CPU is
    the correctness oracle and MPS/cloud is throughput-only (aggregate stats).

Run (from repo root):
    ./packages/eval-tools/scripts/probe_mps_stability.py
"""

from __future__ import annotations

import torch

from eval_tools.policy_spec import parse_policy_spec
from game_runner.runner import run_game
from game_runner.seed_map import list_replay_ids_by_player_count, load_static_from_db
from settings import RUNS_CLOUD_DIR


_CKPT = RUNS_CLOUD_DIR / "2026-05-28T08-21-33Z" / "checkpoints" / "epoch_005.pt"
_SPEC = f"checkpoint:{_CKPT}:force_move=true,value_head_variant=pyramid"
_MAX_TURNS = 200
_N_MAPS = 5


def _play(replay_id: str, device: torch.device):
    static = load_static_from_db(replay_id)
    torch.manual_seed(0)
    policies = [parse_policy_spec(_SPEC, slot=s, device=device) for s in range(2)]
    moves_log: list[list[tuple]] = []

    def on_tick(state, moves, _policies) -> None:  # noqa: ANN001
        moves_log.append([tuple(int(x) for x in m) for m in moves])

    result = run_game(policies, static, max_turns=_MAX_TURNS, on_tick=on_tick)
    return result, moves_log


def _compare(r1, log1, r2, log2) -> dict:
    n = min(len(log1), len(log2))
    first_div = next((t for t in range(n) if log1[t] != log2[t]), None)
    div_ticks = sum(1 for t in range(n) if log1[t] != log2[t])
    return {
        "len": (r1.game_length, r2.game_length),
        "winner": (r1.winner, r2.winner),
        "first_div": first_div,
        "div_ticks": div_ticks,
        "compared": n,
        "land": (
            [ps.land for ps in r1.player_stats],
            [ps.land for ps in r2.player_stats],
        ),
    }


def _run_pair(replay_id: str, device: torch.device) -> dict:
    r1, log1 = _play(replay_id, device)
    r2, log2 = _play(replay_id, device)
    return _compare(r1, log1, r2, log2)


def main() -> int:
    if not _CKPT.exists():
        print(f"checkpoint missing: {_CKPT}")
        return 1
    replay_ids = list_replay_ids_by_player_count(2)[:_N_MAPS]
    if not replay_ids:
        print("no 2-player replay maps in corpus")
        return 1

    print(f"spec: force_move+argmax (no RNG)  max_turns={_MAX_TURNS}")
    print(f"{'map':>10} {'device':>6} {'len r1/r2':>12} {'win r1/r2':>10} "
          f"{'1st_div':>8} {'div/cmp':>10}")

    # CPU control on the first map (should be identical).
    cpu = _run_pair(replay_ids[0], torch.device("cpu"))
    print(f"{replay_ids[0][:10]:>10} {'cpu':>6} "
          f"{f'{cpu['len'][0]}/{cpu['len'][1]}':>12} "
          f"{f'{cpu['winner'][0]}/{cpu['winner'][1]}':>10} "
          f"{str(cpu['first_div']):>8} {f'{cpu['div_ticks']}/{cpu['compared']}':>10}")

    # MPS pairs.
    mps = torch.device("mps")
    for rid in replay_ids:
        c = _run_pair(rid, mps)
        print(f"{rid[:10]:>10} {'mps':>6} "
              f"{f'{c['len'][0]}/{c['len'][1]}':>12} "
              f"{f'{c['winner'][0]}/{c['winner'][1]}':>10} "
              f"{str(c['first_div']):>8} {f'{c['div_ticks']}/{c['compared']}':>10}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
