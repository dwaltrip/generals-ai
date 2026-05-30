"""NNAgent — a generic neural Policy adapter.

Wraps a model "brain" (a shared forward handle + a per-perspective
encoder/decoder) behind the game_runner Policy protocol. NNAgent holds no
knowledge of the model's interiors or the game's mechanics: it extracts a
neutral view via game_runner, hands observations to the brain, and returns the
brain's chosen move. Everything model-specific lives in the brain (e.g.
`bc.inference`'s `BCModelHandle` + `BCPerspective`).

The build_obs / select_action split is the seam a batched runner needs: many
agents' `build_obs` outputs can be stacked into one `forward_batch`, then each
agent's `select_action` consumes its row. `act` composes them for the
single-game path (one batch=1 forward per tick).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from game_types import ObsBundle, PlayerView

from game_runner.sim_adapter import state_to_view


if TYPE_CHECKING:
    import sim_core


class NNAgent:
    def __init__(self, handle: Any, perspective: Any):
        self._handle = handle
        self._perspective = perspective
        self._slot: int = perspective.perspective_slot
        self._pending: ObsBundle | None = None

    def init_for_game(self, state: sim_core.State, static: Any) -> None:
        self._perspective.reset(state_to_view(state, static, self._slot))

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
    def act(self, state: sim_core.State, static: Any) -> tuple[int, int, int]:
        bundle = self.build_obs(state_to_view(state, static, self._slot))
        out_slices = self._handle.forward_batch(bundle.obs[None], bundle.valid_mask[None])
        return self.select_action(out_slices[0])

    # Read by game_runner._build_result via getattr.
    @property
    def n_moved(self) -> int:
        return self._perspective.n_moved

    @property
    def n_passed(self) -> int:
        return self._perspective.n_passed

    def get_diagnostics(self) -> dict[str, Any]:
        return self._perspective.get_diagnostics()
