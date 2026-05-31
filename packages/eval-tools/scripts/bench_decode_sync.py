#!/usr/bin/env -S uv run python
"""De-risk the batched-runner decode path: per-row `.item()` vs. one bulk sync.

The single-game `BCPerspective.decode` interleaves GPU math (argmax, softmax,
topk, sigmoid) with ~9 `.item()` GPU->CPU syncs per call — fine at batch=1, but
a batched runner that calls it per-row would issue B x ~9 syncs/tick, which
could cancel the batching win. This probe measures whether that's real, and how
much a vectorized decode (all math batched, one bulk pull) recovers.

It drives the real `BCModel` forward (random init — fine for timing, like
`bench_scaling.py`) so the decode variants consume real-shaped output, and the
two decode variants replicate the exact ops/sync-count of the production
`decode` + `_record_tick_diagnostics` (force_move + argmax eval path):

  has_legal, value exp-placement / top-prob / entropy, pass sigmoid,
  policy top1 / top3 / entropy, argmax flat_idx  -> ~9 `.item()` reads/row.

Forward latency is reported alongside for context: the question is whether
naive decode is small relative to the forward, or dwarfs it.

Run (from repo root):
    ./packages/eval-tools/scripts/bench_decode_sync.py
"""

from __future__ import annotations

import time

from bc.constants import H_PADDED, OBS_CHANNELS, W_PADDED
from bc.inference import default_device
from bc.loss import flatten_policy_logits
from bc.model import BCModel
import torch


P = 8


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def naive_decode(
    out: dict[str, torch.Tensor], mask: torch.Tensor, device: torch.device,
) -> list[int]:
    """Per-row decode, mirroring `BCPerspective.decode` — ~9 `.item()`s/row."""
    B = out["pass_logit"].shape[0]
    placements = torch.arange(P, device=device, dtype=torch.float32)
    flat_indices: list[int] = []
    for i in range(B):
        policy_logits = out["policy_logits"][i].unsqueeze(0)        # [1, 8, H, W]
        mask_i = mask[i].unsqueeze(0)                               # [1, H, W, 8]
        masked_logits = flatten_policy_logits(policy_logits, mask_i)  # [1, H*W*8]

        has_legal = bool(mask_i.any().item())                      # sync 1

        # value diagnostics
        value_probs = torch.softmax(out["value_logits"][i], dim=0)
        _ = float((value_probs * placements).sum().item()) + 1.0   # sync 2
        _ = float(value_probs.max().item())                        # sync 3
        _ = float(-(value_probs * (value_probs + 1e-12).log()).sum().item())  # sync 4

        # pass head
        _ = float(torch.sigmoid(out["pass_logit"][i]).item())      # sync 5

        # policy diagnostics
        if has_legal:
            probs = torch.softmax(masked_logits[0], dim=0)
            top_k = torch.topk(probs, k=3).values
            _ = float(top_k[0].item())                             # sync 6
            _ = float(top_k.sum().item())                          # sync 7
            _ = float(-(probs * (probs + 1e-12).log()).sum().item())  # sync 8

        flat_indices.append(int(masked_logits.argmax(dim=1).item()))  # sync 9
    return flat_indices


def vec_decode(
    out: dict[str, torch.Tensor], mask: torch.Tensor, device: torch.device,
):
    """Batched decode: same math over [B, ...], one bulk GPU->CPU pull."""
    B = out["pass_logit"].shape[0]
    masked_logits = flatten_policy_logits(out["policy_logits"], mask)  # [B, H*W*8]
    has_legal = mask.reshape(B, -1).any(dim=1)                         # [B]

    value_probs = torch.softmax(out["value_logits"], dim=1)           # [B, 8]
    placements = torch.arange(P, device=device, dtype=value_probs.dtype)
    exp_placement = (value_probs * placements).sum(dim=1) + 1.0       # [B]
    value_top = value_probs.max(dim=1).values
    value_entropy = -(value_probs * (value_probs + 1e-12).log()).sum(dim=1)

    pass_prob = torch.sigmoid(out["pass_logit"])                     # [B]

    probs = torch.softmax(masked_logits, dim=1)                      # [B, H*W*8]
    top_k = torch.topk(probs, k=3, dim=1).values                    # [B, 3]
    top1 = top_k[:, 0]
    top3 = top_k.sum(dim=1)
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=1)
    flat_idx = masked_logits.argmax(dim=1)                          # [B]

    # ONE bulk sync: stack every per-row scalar and pull together.
    bulk = torch.stack(
        [exp_placement, value_top, value_entropy, pass_prob, top1, top3,
         entropy, flat_idx.float(), has_legal.float()],
        dim=1,
    )
    return bulk.cpu().numpy()


def _timeit(fn, *args, device: torch.device, iters: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn(*args)
    _sync(device)
    best = float("inf")
    for _ in range(2):  # two passes, take the better (less noise)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        _sync(device)
        best = min(best, (time.perf_counter() - t0) / iters * 1000)
    return best


def main() -> int:
    device = default_device()
    model = BCModel(value_head_variant="direct").to(device).eval()
    batches = [1, 8, 16, 24, 32, 48]

    print(f"device={device}  value_head=direct  (random init; timing only)")
    print(
        f"{'B':>4} {'fwd_ms':>8} {'naive_ms':>9} {'vec_ms':>8} "
        f"{'naive/fwd':>10} {'speedup':>8} {'us/sync':>8}"
    )
    for B in batches:
        obs = torch.randn(B, OBS_CHANNELS, H_PADDED, W_PADDED, device=device)
        valid = torch.zeros(B, 1, H_PADDED, W_PADDED, dtype=torch.bool, device=device)
        valid[:, :, :20, :20] = True
        # sparse legality mask (~3% true; density doesn't affect dense-op timing)
        mask = torch.rand(B, H_PADDED, W_PADDED, P, device=device) < 0.03

        def _forward():
            with torch.no_grad():
                return model(obs, valid)

        fwd_ms = _timeit(_forward, device=device)

        with torch.no_grad():
            out = model(obs, valid)
        _sync(device)

        naive_ms = _timeit(naive_decode, out, mask, device, device=device)
        vec_ms = _timeit(vec_decode, out, mask, device, device=device)

        us_per_sync = naive_ms * 1000 / (B * 9)
        print(
            f"{B:>4} {fwd_ms:>8.2f} {naive_ms:>9.2f} {vec_ms:>8.2f} "
            f"{naive_ms / fwd_ms:>10.2f} {naive_ms / vec_ms:>8.1f} {us_per_sync:>8.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
