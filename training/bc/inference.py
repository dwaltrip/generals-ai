"""BC model inference — turn a trained BCModel into a game-playing brain.

This is the model's *live inference contract*, kept next to the model and the
observation encoder it depends on. It consumes the neutral
`game_types.PlayerView` (raw per-tick game arrays) rather than `sim_core.State`,
so this module — and therefore `training/bc` — has no dependency on the game
engine. The `State -> PlayerView` translation lives in `game-runner`.

Two pieces, split at the forward-pass seam so a future vectorized runner can
batch forwards across many perspectives:

  - `BCModelHandle`: shared per checkpoint. Owns the model + device and runs
    the (batched) forward. `forward_batch` runs batch=1 today.
  - `BCPerspective`: one (game, slot). Owns the growing sim dict, `MemoryState`,
    `BFSCache`, decode options, and per-tick diagnostics. `encode(view)` builds
    the obs tensor + masks (CPU numpy); `decode(out_slice)` selects the move.

Fog-of-war and the 8-slot general padding are applied here, exactly as the
pre-refactor `ModelAgent` did — behavior is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from game_types import ObsBundle, PlayerView
import numpy as np
import torch

from bc import actions, visibility
from bc import bfs as bc_bfs
from bc import mask as bc_mask
from bc import obs as bc_obs
from bc.checkpoint import load_bc_model
from bc.constants import H_PADDED, W_PADDED
from bc.loss import flatten_policy_logits
from bc.model import BCModel
from bc.obs import pad_initial_generals


# Fixed slot count the model + obs encoder were trained on (8-player FFA).
P = 8


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# A single perspective's slice of one batched forward: the model output dict
# indexed to one row (no batch dim). Opaque to the generic agent; produced and
# consumed within this module.
OutSlice = dict[str, torch.Tensor]


class BCModelHandle:
    """The model + device, shared across every perspective that uses this
    checkpoint. `model_key` lets a batched runner group perspectives that can
    share one forward.
    """

    def __init__(self, model: BCModel, device: torch.device, model_key: str):
        self.model = model
        self.device = device
        self.model_key = model_key

    @classmethod
    def load(
        cls, path: str | Path, device: torch.device, value_head_variant: str = "direct",
    ) -> BCModelHandle:
        model = load_bc_model(path, device, value_head_variant)
        return cls(model, device, model_key=f"{path}|{device}|{value_head_variant}")

    def forward_batch(
        self, obs_batch: np.ndarray, valid_batch: np.ndarray,
    ) -> list[OutSlice]:
        """Run one forward over a stacked batch and return per-row output slices.

        obs_batch:   float32 [B, OBS_CHANNELS, H_PADDED, W_PADDED]
        valid_batch: bool    [B, 1, H_PADDED, W_PADDED]
        """
        obs_t = torch.from_numpy(obs_batch).to(self.device)
        valid_t = torch.from_numpy(valid_batch).to(self.device)
        with torch.no_grad():
            out = self.model(obs_t, valid_t)
        return [
            {
                "policy_logits": out["policy_logits"][i],  # [8, H, W]
                "pass_logit": out["pass_logit"][i],        # scalar
                "value_logits": out["value_logits"][i],    # [8]
            }
            for i in range(obs_batch.shape[0])
        ]


class BCPerspective:
    """One perspective's live state. Stateless across games — re-init via
    `reset(...)` each new game. Holds the growing sim snapshots, running memory,
    BFS cache, and decode options.

    `force_move`: ignore the pass head when at least one move is legal under the
        mask (the BC pass head is biased toward "do nothing").
    `sample`: draw from softmax(masked policy logits) instead of argmax;
        `temperature` scales the softmax.
    """

    def __init__(
        self,
        perspective_slot: int,
        device: torch.device,
        *,
        force_move: bool = False,
        sample: bool = False,
        temperature: float = 1.0,
    ):
        self.perspective_slot = perspective_slot
        self.device = device
        self.opp_slots = bc_obs.canonical_slot_order(perspective_slot, P)[1:]
        self.force_move = force_move
        self.sample = sample
        self.temperature = temperature

        # Filled by reset()
        self._H: int = 0
        self._W: int = 0
        self._ownership: list[np.ndarray] | None = None
        self._armies: list[np.ndarray] | None = None
        self._sim: dict[str, Any] | None = None
        self._memory: bc_obs.MemoryState | None = None
        self._bfs_cache: bc_bfs.BFSCache | None = None

        # Per-game decision counters + per-tick diagnostic series.
        self.n_passed = 0
        self.n_moved = 0
        self.n_no_legal = 0
        self._value_exp_placement_per_tick: list[float] = []
        self._value_top_prob_per_tick: list[float] = []
        self._value_entropy_per_tick: list[float] = []
        self._pass_prob_per_tick: list[float] = []
        self._top1_prob_per_tick: list[float] = []
        self._top3_prob_per_tick: list[float] = []
        self._entropy_per_tick: list[float] = []

    def reset(self, view: PlayerView) -> None:
        """Initialize per-game state from the t=0 view (was `init_for_game`)."""
        H, W = view.H, view.W
        self._H, self._W = H, W
        self._ownership = [np.asarray(view.ownership_t, dtype=np.int8)]
        self._armies = [np.asarray(view.armies_t, dtype=np.int16)]

        # initial_generals is length num_players; the BC encoder was trained on
        # 8-player FFA and unconditionally indexes slots 0..P-1. Pad unused slots
        # with the perspective's own general cell (idempotent under fancy
        # indexing; never triggers a false sighting in step_memory).
        ig_padded = pad_initial_generals(view.initial_generals, self.perspective_slot)

        self._sim = {
            "ownership": self._ownership,
            "armies": self._armies,
            "mountains": np.asarray(view.mountains, dtype=np.int32),
            "initial_cities": np.asarray(view.initial_cities, dtype=np.int32),
            "initial_generals": ig_padded,
            "cities": np.asarray(view.cities, dtype=np.int32),
            "cities_present_at": np.asarray(view.cities_present_at, dtype=np.int32),
            "capture_events": np.asarray(view.capture_events, dtype=np.int32),
        }
        self._memory = bc_obs.init_memory_live(self._sim, self.perspective_slot, H, W, P)
        self._bfs_cache = bc_bfs.init_bfs_cache(P)

        self.n_passed = 0
        self.n_moved = 0
        self.n_no_legal = 0
        self._value_exp_placement_per_tick = []
        self._value_top_prob_per_tick = []
        self._value_entropy_per_tick = []
        self._pass_prob_per_tick = []
        self._top1_prob_per_tick = []
        self._top3_prob_per_tick = []
        self._entropy_per_tick = []

    def encode(self, view: PlayerView) -> ObsBundle:
        """Advance memory and build the obs tensor + masks for tick t.

        Mirrors steps 1-6 of the old `ModelAgent.act`, reading from the neutral
        `PlayerView` instead of `sim_core.State`.
        """
        assert self._sim is not None, "call reset() first"
        assert self._memory is not None
        assert self._bfs_cache is not None
        assert self._ownership is not None
        assert self._armies is not None

        t = view.timestep
        H, W = self._H, self._W

        # 1. Append latest snapshot row.
        own_t = np.asarray(view.ownership_t, dtype=np.int8)
        arm_t = np.asarray(view.armies_t, dtype=np.int16)
        self._ownership.append(own_t)
        self._armies.append(arm_t)

        # 2. Refresh dynamic sim fields (cities + captures grow mid-game).
        self._sim["cities"] = np.asarray(view.cities, dtype=np.int32)
        self._sim["cities_present_at"] = np.asarray(view.cities_present_at, dtype=np.int32)
        self._sim["capture_events"] = np.asarray(view.capture_events, dtype=np.int32)

        # 3. Append scoreboard row for tick t.
        land, army = bc_obs.scoreboard_row(own_t, arm_t, P)
        self._memory.land_count_history.append(land)
        self._memory.army_count_history.append(army)

        # 4. Visibility for this perspective.
        vis = visibility.compute_visibility(own_t, self.perspective_slot, H, W)

        # 5. Advance memory; invalidate BFS cache if the known-passable graph grew.
        graph_grew = bc_obs.step_memory(
            self._memory, self._sim, t, vis, self.perspective_slot, H, W, P,
        )
        if graph_grew:
            self._bfs_cache.invalidate_graph()

        # 6. Build obs + legality mask + valid-region mask.
        obs_np = bc_obs.build_obs(
            self._sim, t, self.perspective_slot, self.opp_slots,
            vis, self._memory, self._bfs_cache, H, W,
        )
        mask_np = bc_mask.build_mask(self._sim, t, self.perspective_slot, H, W)
        valid_np = np.zeros((1, H_PADDED, W_PADDED), dtype=bool)
        valid_np[0, :H, :W] = True

        return ObsBundle(obs=obs_np, policy_mask=mask_np, valid_mask=valid_np)

    def decode(self, out_slice: OutSlice, policy_mask: np.ndarray) -> tuple[int, int, int]:
        """Select a wire move `(source, dest, is50)` from this perspective's
        model output. `(-1, -1, -1)` means pass. Mirrors step 7 of the old
        `ModelAgent.act` plus its per-tick diagnostics.
        """
        W = self._W
        # flatten_policy_logits wants a batch dim on both logits and mask.
        policy_logits = out_slice["policy_logits"].unsqueeze(0)            # [1, 8, H, W]
        mask_t = torch.from_numpy(policy_mask).unsqueeze(0).to(self.device)  # [1, H, W, 8]
        masked_logits = flatten_policy_logits(policy_logits, mask_t)      # [1, H*W*8]

        has_legal = bool(mask_t.any())
        self._record_tick_diagnostics(out_slice, masked_logits, has_legal)

        if not has_legal:
            # No legal move (e.g. very early game, all owned tiles army<2). Pass.
            self.n_no_legal += 1
            return -1, -1, -1

        # Pass decision (unless force_move): sample bernoulli(sigmoid(logit))
        # under `sample`, else hard threshold logit > 0.
        if not self.force_move:
            if self.sample:
                p_pass = torch.sigmoid(out_slice["pass_logit"])
                pass_head_says_pass = bool(torch.bernoulli(p_pass).item())
            else:
                pass_head_says_pass = bool((out_slice["pass_logit"] > 0).item())
            if pass_head_says_pass:
                self.n_passed += 1
                return -1, -1, -1

        if self.sample:
            probs = torch.softmax(masked_logits / self.temperature, dim=1)
            flat_idx = int(torch.multinomial(probs, num_samples=1).item())
        else:
            flat_idx = int(masked_logits.argmax(dim=1).item())

        self.n_moved += 1
        return actions.decode(
            is_pass=False, flat_idx=flat_idx, W_unpadded=W, W_padded=W_PADDED,
        )

    def _record_tick_diagnostics(
        self, out_slice: OutSlice, masked_logits: torch.Tensor, has_legal: bool,
    ) -> None:
        # Value head: categorical placement over 8 classes (0=1st, 7=8th).
        value_probs = torch.softmax(out_slice["value_logits"], dim=0)
        placements = torch.arange(
            value_probs.size(0), dtype=value_probs.dtype, device=value_probs.device,
        )
        exp_placement = float((value_probs * placements).sum().item()) + 1.0  # 1..8
        top_prob = float(value_probs.max().item())
        value_entropy = float(-(value_probs * (value_probs + 1e-12).log()).sum().item())
        self._value_exp_placement_per_tick.append(exp_placement)
        self._value_top_prob_per_tick.append(top_prob)
        self._value_entropy_per_tick.append(value_entropy)

        # Pass head: sigmoid of scalar logit.
        self._pass_prob_per_tick.append(float(torch.sigmoid(out_slice["pass_logit"]).item()))

        # Policy head: only meaningful when at least one legal slot exists.
        if has_legal:
            probs = torch.softmax(masked_logits[0], dim=0)
            k = min(3, probs.size(0))
            top_k = torch.topk(probs, k=k).values
            self._top1_prob_per_tick.append(float(top_k[0].item()))
            self._top3_prob_per_tick.append(float(top_k.sum().item()))
            self._entropy_per_tick.append(
                float(-(probs * (probs + 1e-12).log()).sum().item())
            )

    def get_diagnostics(self) -> dict[str, Any]:
        """Per-game diagnostic summary (picked up by `MetricsCollector` via
        duck-typed `getattr(policy, "get_diagnostics", None)`)."""
        def _agg(xs: list[float]) -> dict[str, float]:
            if not xs:
                return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
            arr = np.asarray(xs)
            return {
                "mean": float(arr.mean()),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
            }

        ep = self._value_exp_placement_per_tick
        if ep:
            arr = np.asarray(ep)
            value_pos = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        else:
            value_pos = {k: float("nan") for k in ("mean", "std", "min", "max")}

        low_conf = sum(1 for p in self._top1_prob_per_tick if p < 0.3)
        high_conf = sum(1 for p in self._top1_prob_per_tick if p > 0.8)

        return {
            "value": {
                "exp_placement": {**value_pos, "per_tick": ep},
                "top_prob": {**_agg(self._value_top_prob_per_tick),
                             "per_tick": self._value_top_prob_per_tick},
                "entropy": {**_agg(self._value_entropy_per_tick),
                            "per_tick": self._value_entropy_per_tick},
            },
            "pass_prob": {
                **_agg(self._pass_prob_per_tick),
                "per_tick": self._pass_prob_per_tick,
            },
            "policy": {
                "top1_prob": {**_agg(self._top1_prob_per_tick),
                              "per_tick": self._top1_prob_per_tick},
                "top3_prob_mean": (
                    float(np.mean(self._top3_prob_per_tick))
                    if self._top3_prob_per_tick else float("nan")
                ),
                "entropy": {**_agg(self._entropy_per_tick),
                            "per_tick": self._entropy_per_tick},
                "low_conf_ticks": low_conf,
                "high_conf_ticks": high_conf,
            },
            "decisions": {
                "n_moved": self.n_moved,
                "n_passed": self.n_passed,
                "n_no_legal": self.n_no_legal,
            },
            "config": {
                "force_move": self.force_move,
                "sample": self.sample,
                "temperature": self.temperature,
            },
        }
