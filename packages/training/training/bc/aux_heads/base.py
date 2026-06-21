"""The aux-head spec interface: one object per variant-gated auxiliary head.

An aux head (today: the two elimination variants) needs the same handful of
things at every stage of the pipeline — a head module to build, per-frame
targets to encode, a loss term, dump records, and in-loop eval diagnostics. A
*spec* bundles those so the pipeline dispatches through one registry lookup
instead of a per-site `if variant == ...` branch. The contract is head-neutral:
nothing here names "elim".

See `docs/2026-06/6.21-1-aux-head-spec-registry-refactor.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn


@dataclass
class AuxLossResult:
    """What a spec's `loss` returns — everything `bc_loss` needs to fold the head
    into the total and the reporting dict, without `bc_loss` knowing the variant.

    `metrics` holds every reported quantity by name (e.g. the hard reporting CE
    and the soft objective). `term_key` selects which of them is the *trained*
    objective — the scalar that enters `total` (weighted by `weight`); it is also
    a reported metric, so the τ=0 "soft == hard" guarantee stays a property of the
    metrics dict. `count` is the head's own denominator (alive-pair count, defined-
    next-victim count, … — NOT the sample count), reported under `count_key` so the
    accumulator can weight the running means by it.
    """

    metrics: dict[str, torch.Tensor]
    term_key: str
    weight: float
    count: torch.Tensor
    count_key: str


@runtime_checkable
class AuxHeadSpec(Protocol):
    """The per-variant interface. Implementations are stateless singletons held in
    the registry; all run-specific data (arch, loss cfg, edges, the per-game ctx)
    is passed in per call.

    The methods split across three runtime contexts — numpy in the dataloader
    workers (`encode_targets`), torch on the model/train side
    (`build_head`/`loss`/`dump_records`), and the eval loop (`build_eval_meter`/
    `eval_update`/`eval_summary`). Head-class and meter imports are lazy inside the
    methods that need them so importing a spec from a worker stays numpy-light.

    The per-game `ElimCtx` precompute is *not* here: it is variant-independent
    (carries both death and removal markers) and shared with the standalone
    alive-mask path, so it lives once in the dataset walk, not per spec.
    """

    name: str
    output_key: str

    def build_head(self, arch: Any) -> nn.Module: ...

    def encode_targets(
        self, ctx: Any, raw_order: list[int], t: int
    ) -> dict[str, torch.Tensor]: ...

    def loss(
        self,
        model_out: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        cfg: Any,
    ) -> AuxLossResult: ...

    def dump_records(
        self,
        out: dict[str, torch.Tensor],
        moved_batch: dict[str, torch.Tensor],
        host_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]: ...

    def build_eval_meter(self, cfg: Any, edges: tuple[int, ...] | None) -> Any | None: ...

    def eval_update(
        self,
        meter: Any,
        out: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> None: ...

    def eval_summary(
        self, meter: Any, accum_summary: dict[str, float | int]
    ) -> dict[str, float | None]: ...

    def validate(self, arch: Any, cfg: Any) -> None: ...
