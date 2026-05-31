#!/usr/bin/env -S uv run python
"""De-risk: is the forward batch-invariant? forward(B=1) vs row i of forward(B=N).

The model is GroupNorm, so rows are *algebraically* independent — batching is
mathematically identical to running each row alone. This probe checks whether
that holds at the bit level on each device, which is the precondition the
multi-game oracle's strictness hinges on:

  - CPU bitwise-identical  -> oracle can compare diagnostics *exactly*.
  - CPU only allclose       -> oracle compares diagnostics with a tolerance
                               (moves/argmax stay exact — argmax shrugs off ULP).
  - MPS: the magnitude of the cross-batch logit delta, and whether it ever flips
    an argmax, is the real "does batched == sequential on MPS" answer that the
    run-to-run stability probe left open.

Uses real obs (captured by stepping a short NN-vs-NN game) so the argmax-flip
check sees realistic near-ties, not random noise.

Run (from repo root):
    ./packages/eval-tools/scripts/probe_row_independence.py
"""

from __future__ import annotations

from pathlib import Path

from bc.inference import BCModelHandle, BCPerspective
from bc.loss import flatten_policy_logits
from game_runner.sim_adapter import state_to_view
from game_runner.seed_map import list_replay_ids_by_player_count, load_static_from_db
import numpy as np
import torch

import sim_core


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CKPT = _REPO_ROOT / "training/data/runs-cloud/2026-05-28T08-21-33Z/checkpoints/epoch_005.pt"
_STEPS = 30
_BATCH = 32


def _collect_real_obs(handle: BCModelHandle, device: torch.device):
    """Step a short NN-vs-NN game, capturing each perspective's obs bundle."""
    rid = list_replay_ids_by_player_count(2)[0]
    static = load_static_from_db(rid)
    state = sim_core.new_state(static)
    persps = [BCPerspective(p, device, force_move=True) for p in (0, 1)]
    for p in persps:
        p.reset(state_to_view(state, static, p.perspective_slot))

    bundles = []
    for _ in range(_STEPS):
        if state.alive_count <= 1:
            break
        moves = []
        for p, persp in enumerate(persps):
            if not state.alive[p]:
                continue
            view = state_to_view(state, static, p)
            bundle = persp.encode(view)
            bundles.append(bundle)
            out = handle.forward_batch(bundle.obs[None], bundle.valid_mask[None])
            move = persp.decode(out[0], bundle.policy_mask)
            if move[0] != -1:
                moves.append((p, *move))
        state.step_tick(moves=moves, afks=[])
    return bundles


def _argmax_idx(out_slice, policy_mask, device: torch.device) -> tuple[int, float]:
    """Masked-policy argmax index + the top1−top2 logit gap (for context)."""
    pl = out_slice["policy_logits"].unsqueeze(0)
    m = torch.from_numpy(policy_mask).unsqueeze(0).to(device)
    masked = flatten_policy_logits(pl, m)[0]
    top2 = torch.topk(masked, k=2).values
    gap = float((top2[0] - top2[1]).item())
    return int(masked.argmax().item()), gap


def _run_device(name: str) -> None:
    device = torch.device(name)
    handle = BCModelHandle.load(_CKPT, device, value_head_variant="pyramid")
    bundles = _collect_real_obs(handle, device)
    n = min(_BATCH, len(bundles))
    batch = bundles[:n]

    obs_stack = torch.from_numpy(np.stack([b.obs for b in batch])).to(device)
    valid_stack = torch.from_numpy(np.stack([b.valid_mask for b in batch])).to(device)
    with torch.no_grad():
        out_batched = handle.model(obs_stack, valid_stack)
    batched_rows = [
        {
            "policy_logits": out_batched["policy_logits"][i],
            "pass_logit": out_batched["pass_logit"][i],
            "value_logits": out_batched["value_logits"][i],
        }
        for i in range(n)
    ]

    max_dp = max_dpass = max_dv = 0.0
    argmax_disagree = pass_disagree = 0
    gaps = []
    for i, b in enumerate(batch):
        solo = handle.forward_batch(b.obs[None], b.valid_mask[None])[0]
        bat = batched_rows[i]
        max_dp = max(max_dp, float((bat["policy_logits"] - solo["policy_logits"]).abs().max().item()))
        max_dpass = max(max_dpass, float((bat["pass_logit"] - solo["pass_logit"]).abs().item()))
        max_dv = max(max_dv, float((bat["value_logits"] - solo["value_logits"]).abs().max().item()))

        ai_b, gap = _argmax_idx(bat, b.policy_mask, device)
        ai_s, _ = _argmax_idx(solo, b.policy_mask, device)
        gaps.append(gap)
        if ai_b != ai_s:
            argmax_disagree += 1
        if bool((bat["pass_logit"] > 0).item()) != bool((solo["pass_logit"] > 0).item()):
            pass_disagree += 1

    med_gap = sorted(gaps)[len(gaps) // 2] if gaps else float("nan")
    print(f"\n=== device={name}  rows={n} ===")
    print(f"  max|Δ| logits:  policy={max_dp:.3e}  pass={max_dpass:.3e}  value={max_dv:.3e}")
    print(f"  argmax disagreements:  {argmax_disagree}/{n}")
    print(f"  pass-decision disagreements:  {pass_disagree}/{n}")
    print(f"  median top1−top2 policy-logit gap:  {med_gap:.3e}  (Δ « gap ⇒ no flips)")
    exact = max_dp == 0.0 and max_dpass == 0.0 and max_dv == 0.0
    print(f"  bitwise batch-invariant: {'YES' if exact else 'NO (allclose only)'}")


def main() -> int:
    if not _CKPT.exists():
        print(f"checkpoint missing: {_CKPT}")
        return 1
    _run_device("cpu")
    if torch.backends.mps.is_available():
        _run_device("mps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
