"""NNAgent — a generic neural Policy adapter.

Wraps a model "brain" (a shared forward handle + a per-perspective
encoder/decoder) behind the game_runner Policy protocol. NNAgent holds no
knowledge of the model's interiors or the game's mechanics: it extracts a
neutral view via game_runner, hands observations to the brain, and returns the
brain's chosen move. Everything model-specific lives in the brain (e.g.
`bc.inference`'s `BCModelHandle` + `BCPerspective`). The `ModelHandle` and
`Perspective` protocols below name exactly what a brain must provide.

The build_obs / select_action split is the seam a batched runner needs: many
agents' `build_obs` outputs can be stacked into one `forward_batch`, then each
agent's `select_action` consumes its row. `act` composes them for the
single-game path (one batch=1 forward per tick).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from game_types import ObsBundle, PlayerView, StaticMap

from game_runner.sim_adapter import state_to_view


if TYPE_CHECKING:
    import numpy as np
    import sim_core


class ModelHandle(Protocol):
    """The model side of a brain: runs a (possibly batched) forward over
    stacked observations and returns one opaque output slice per input row.
    NNAgent never inspects a slice — it only routes it back to the matching
    perspective's `decode`.
    """

    def forward_batch(
        self, obs_batch: np.ndarray, valid_batch: np.ndarray,
    ) -> list[Any]: ...


class Perspective(Protocol):
    """The per-(game, slot) side of a brain. `encode` builds the observation
    the handle consumes; `decode` turns the handle's output slice back into a
    wire move. The counters and `get_diagnostics` are read by game_runner when
    it assembles the game result.
    """

    perspective_slot: int

    def reset(self, view: PlayerView) -> None: ...
    def encode(self, view: PlayerView) -> ObsBundle: ...
    def decode(self, out_slice: Any, policy_mask: np.ndarray) -> tuple[int, int, int]: ...

    n_moved: int
    n_passed: int
    n_no_legal: int

    def get_diagnostics(self) -> dict[str, Any]: ...


class NNAgent:
    def __init__(self, handle: ModelHandle, perspective: Perspective):
        self._handle = handle
        self._perspective = perspective
        self._slot: int = perspective.perspective_slot
        self._pending: ObsBundle | None = None

    def init_for_game(self, state: sim_core.State, map_data: StaticMap) -> None:
        self._perspective.reset(state_to_view(state, map_data, self._slot))

    # --- batchable interface: build observations / consume model output ---
    def build_obs(self, view: PlayerView) -> ObsBundle:
        bundle: ObsBundle = self._perspective.encode(view)
        self._pending = bundle
        return bundle

    def select_action(self, out_slice: Any) -> tuple[int, int, int]:
        assert self._pending is not None, "build_obs must precede select_action"
        move = self._perspective.decode(out_slice, self._pending.policy_mask)
        self._pending = None
        return move

    # --- single-game Policy.act: compose build -> forward(batch=1) -> select ---
    def act(self, state: sim_core.State, map_data: StaticMap) -> tuple[int, int, int]:
        bundle = self.build_obs(state_to_view(state, map_data, self._slot))
        out_slices = self._handle.forward_batch(bundle.obs[None], bundle.valid_mask[None])
        return self.select_action(out_slices[0])

    # Read by game_runner._build_result via getattr.
    @property
    def n_moved(self) -> int:
        return self._perspective.n_moved

    @property
    def n_passed(self) -> int:
        return self._perspective.n_passed

    @property
    def n_no_legal(self) -> int:
        return self._perspective.n_no_legal

    def get_diagnostics(self) -> dict[str, Any]:
        return self._perspective.get_diagnostics()
