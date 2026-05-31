"""Protocols for a batchable model "brain" that a runner can drive generically.

These let `game_runner` orchestrate a batched forward across many perspectives
without importing torch or any model package: the runner shuttles *opaque*
forward-outputs (and, later, decode-configs) between a handle and routes each
row back to the producing policy. Concrete implementations live elsewhere
(`bc.inference`'s `BCModelHandle`, `self_play`'s `NNAgent`) and satisfy these
structurally — no import of this module required to conform.

A `Policy` (see `policy.py`) that also satisfies `BatchablePolicy` can have its
per-tick forward stacked with others'; a plain `Policy` (e.g. a CPU bot) is
driven one tick at a time via `act`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from game_types import ObsBundle, PlayerView
    import numpy as np


class ModelHandle(Protocol):
    """The model side of a brain, shared across every perspective on one
    checkpoint. `model_key` lets a runner group perspectives whose forwards can
    be merged into one batch.
    """

    model_key: str

    def forward_batch(self, obs_batch: np.ndarray, valid_batch: np.ndarray) -> Any: ...
    def decode_batch(
        self, out: Any, mask_batch: np.ndarray, configs: list[Any],
    ) -> list[Any]: ...


class BatchablePolicy(Protocol):
    """A `Policy` whose per-tick decision splits around a (batchable) forward:
    `build_obs` produces the observation to stack, `select_action` consumes the
    matching decoded row, and `decode_config` supplies the per-row options the
    handle's `decode_batch` branches on. A runner detects these by duck-typing —
    a `Policy` without `build_obs` is treated as a CPU policy and driven via `act`.
    """

    model_handle: ModelHandle
    decode_config: Any

    def build_obs(self, view: PlayerView) -> ObsBundle: ...
    def select_action(self, decision: Any) -> tuple[int, int, int]: ...
