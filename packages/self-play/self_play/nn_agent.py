"""NNAgent — a generic neural Policy adapter.

Wraps a model "brain" (a shared forward handle + a per-perspective
encoder/decoder) behind the game_runner Policy protocol. NNAgent holds no
knowledge of the model's interiors or the game's mechanics: it extracts a
neutral view via game_runner, hands observations to the brain, and returns the
brain's chosen move. Everything model-specific lives in the brain (e.g.
`bc.inference`'s `BCModelHandle` + `BCPerspective`). `ModelHandle` (in
`game_runner.brain`) and the `Perspective` protocol below name exactly what a
brain must provide.

The build_obs / select_action split is the seam a batched runner needs: many
agents' `build_obs` outputs can be stacked into one `forward_batch`, then each
agent's `select_action` consumes its row. `act` composes them for the
single-game path (one batch=1 forward per tick).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from game_types import ObsBundle, PlayerView, StaticMap

from game_runner.brain import ModelHandle
from game_runner.sim_adapter import state_to_view


if TYPE_CHECKING:
    import sim_core


class Perspective(Protocol):
    """The per-(game, slot) side of a brain. `encode` builds the observation the
    handle consumes; `select_action` turns the handle's decoded row back into a
    wire move and records this tick's diagnostics; `decode_config` supplies the
    per-row options the handle's `decode_batch` branches on. The counters and
    `get_diagnostics` are read by game_runner when it assembles the game result.
    """

    perspective_slot: int
    decode_config: Any

    def reset(self, view: PlayerView) -> None: ...
    def encode(self, view: PlayerView) -> ObsBundle: ...
    def select_action(self, decision: Any) -> tuple[int, int, int]: ...

    n_moved: int
    n_passed: int
    n_no_legal: int

    def get_diagnostics(self) -> dict[str, Any]: ...


class NNAgent:
    def __init__(self, handle: ModelHandle, perspective: Perspective):
        self._handle = handle
        self._perspective = perspective
        self._slot: int = perspective.perspective_slot

    @property
    def model_handle(self) -> ModelHandle:
        return self._handle

    @property
    def decode_config(self) -> Any:
        return self._perspective.decode_config

    def init_for_game(self, state: sim_core.State, map_data: StaticMap) -> None:
        self._perspective.reset(state_to_view(state, map_data, self._slot))

    # --- batchable interface: a runner stacks build_obs across slots, runs one
    # forward_batch + decode_batch per model, then routes each row here. ---
    def build_obs(self, view: PlayerView) -> ObsBundle:
        return self._perspective.encode(view)

    def select_action(self, decision: Any) -> tuple[int, int, int]:
        return self._perspective.select_action(decision)

    # --- single-game Policy.act: the batched spine at batch-of-1. ---
    def act(self, state: sim_core.State, map_data: StaticMap) -> tuple[int, int, int]:
        bundle = self.build_obs(state_to_view(state, map_data, self._slot))
        out = self._handle.forward_batch(bundle.obs[None], bundle.valid_mask[None])
        decisions = self._handle.decode_batch(
            out, bundle.policy_mask[None], [self._perspective.decode_config],
        )
        return self.select_action(decisions[0])

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
