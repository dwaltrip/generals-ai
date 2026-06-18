"""The A–E scorer families and the `ArgminModel` wrapper.

A scorer maps the encoded frame `x = [B, 8, C]` to `[B, 8]` logits; `ArgminModel`
glues an `EncoderCfg` to a scorer so one cached table feeds every variant (the
encoder runs inside `forward`). The decision is `argmax` with the lowest-index
tiebreak — `torch.argmax` returns the first maximal index, matching the `label`
rule (continuous logits ~never tie exactly; the genuine ambiguity is in the label).

The scorers trace two things at once (6.18-1 §4):
  - depth: A1 (linear, depth-0) → B1 (one hidden layer, the minimal notch-capable
    model). B1 at depth 0 *is* A1.
  - the equivariance spine A1 → D1 → E1: unconstrained linear → permutation-
    equivariant linear → context-free shared-φ (the rung that bridges to the real
    aux head). All share A1's per-player scalars; D1/E1 are tiny.

Reference solutions are stated for `C=1` (army-only) in normalized units (the
encoder's `/1000` runs inside `forward`): A1 `W ≈ −I`, D1 `(A,B) ≈ (−1,0)`, E1
`w ≈ −1`. Each scorer exposes `.inspect()` so weight readout is per-model rather
than the driver reaching into internals — "the matrix" only exists for A1 at `C=1`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from training.analysis.argmin_toy.encode import EncoderCfg, encode


N_PLAYERS = 8


def _sym_subspace_residual(W: torch.Tensor) -> float:
    """Relative Frobenius residual of `W` after the least-squares fit to the
    permutation-symmetric subspace `a·I + b·11ᵀ` (diagonal `a+b`, off-diagonal `b`).
    `b = off-diagonal mean`, `a = diagonal mean − b`. ~0 when `W ≈ −I`. Defined for
    the square `[8,8]` case (A1 at `C=1`) only — callers pass `None` otherwise."""
    n = W.shape[0]
    diag = torch.diagonal(W)
    off_sum = W.sum() - diag.sum()
    b = off_sum / (n * n - n)
    a = diag.mean() - b
    fit = a * torch.eye(n, dtype=W.dtype) + b
    denom = W.norm().clamp(min=1e-12)
    return float((W - fit).norm() / denom)


class A1(nn.Module):
    """Unconstrained pool-first linear: flatten `[B, 8·C]` → `Linear(8·C, 8)`. The
    depth-0 anchor; reference `W ≈ −I` at `C=1`."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.lin = nn.Linear(N_PLAYERS * in_ch, N_PLAYERS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x.reshape(x.shape[0], -1))

    def inspect(self) -> dict:
        W = self.lin.weight.detach()  # [8, 8·C]
        res = _sym_subspace_residual(W) if self.in_ch == 1 else None
        return {"W": W, "sym_subspace_residual": res}


class B1(nn.Module):
    """Pooled MLP, depth-1: flatten → `Linear(8·C, hidden) → ReLU → Linear(hidden, 8)`.
    The minimal notch-capable model; no clean `W`, so `.inspect()` is empty."""

    def __init__(self, in_ch: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_PLAYERS * in_ch, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_PLAYERS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))

    def inspect(self) -> dict:
        return {}


class D1(nn.Module):
    """Equivariant linear: `score_p = A·x_p + B·Σ_q x_q`, `2C` params. Permutation-
    equivariant by construction; reference `(A, B) ≈ (−1, 0)` at `C=1`."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_ch) * 0.01)
        self.B = nn.Parameter(torch.randn(in_ch) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        own = x @ self.A                       # [B, 8]
        ctx = (x.sum(dim=1) @ self.B)          # [B]
        return own + ctx.unsqueeze(1)          # broadcast the shared context term

    def inspect(self) -> dict:
        return {"A": self.A.detach(), "B": self.B.detach()}


class E1(nn.Module):
    """Shared-φ: `φ(x_p)` applied per player, φ shared (context-free). Linear φ →
    `w ≈ −1` at `C=1`; a tiny MLP φ (one hidden layer) for the notch encodings."""

    def __init__(self, in_ch: int, phi_hidden: int | None = None):
        super().__init__()
        self.linear_phi = phi_hidden is None
        if phi_hidden is None:
            self.phi: nn.Module = nn.Linear(in_ch, 1)
        else:
            self.phi = nn.Sequential(
                nn.Linear(in_ch, phi_hidden), nn.ReLU(), nn.Linear(phi_hidden, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.phi(x).squeeze(-1)         # [B, 8, 1] -> [B, 8]

    def inspect(self) -> dict:
        # Linear φ has a readable weight (`w ≈ −1`); an MLP φ does not.
        w = self.phi.weight.detach().flatten() if self.linear_phi else None  # type: ignore[union-attr]
        return {"w": w}


class ArgminModel(nn.Module):
    """An `EncoderCfg` + a scorer. `forward(army, land, alive)` encodes then scores
    to `[B, 8]` logits; `land` is threaded through from the start (unused under
    `army_only`, wired for the `+land` rider). `.inspect()` delegates to the scorer."""

    def __init__(self, encoder_cfg: EncoderCfg, scorer: nn.Module):
        super().__init__()
        self.encoder_cfg = encoder_cfg
        self.scorer = scorer

    def forward(
        self, army: torch.Tensor, land: torch.Tensor, alive: torch.Tensor
    ) -> torch.Tensor:
        return self.scorer(encode(army, land, alive, self.encoder_cfg))

    def inspect(self) -> dict:
        return self.scorer.inspect()  # type: ignore[attr-defined]
