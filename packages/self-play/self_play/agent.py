"""ModelAgent — one perspective's brain for a self-play game.

Holds the model, per-player MemoryState / BFSCache, a growing `sim` dict
(ownership/armies lists appended per tick), and the canonical-slot
ordering for this perspective.

The act() loop on each tick:
  1. Append the latest sim snapshot to the growing lists
  2. Refresh dynamic sim fields (cities, capture_events)
  3. Append scoreboard row for the current tick
  4. Compute the perspective's visibility
  5. step_memory → build_obs → build_mask
  6. Model forward, masked argmax over policy logits, pass-head sign
  7. actions.decode → wire move (source, dest, is50)

Two ModelAgent instances drive a 2-player game; they share the same
underlying nn.Module (same weights = self-play). Each instance carries
its own perspective state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

import sim_core

from bc import actions, bfs as bc_bfs, mask as bc_mask, obs as bc_obs, visibility
from bc.constants import H_PADDED, W_PADDED
from bc.loss import flatten_policy_logits
from bc.model import BCModel

from game_runner import sim_adapter


P = 8  # fixed slot count the model + obs encoder were trained on


def pad_initial_generals(
    initial_generals, perspective_slot: int, num_slots: int = P,
) -> np.ndarray:
    """Pad initial_generals to `num_slots` length for the obs encoder.

    The BC encoder indexes slots 0..P-1 unconditionally. For <P-player
    games, unused slots are filled with the perspective's own general cell
    — idempotent under fancy indexing and safe in step_memory's per-player
    loop (own general is already known, so padded slots never trigger
    false sightings).
    """
    ig = np.asarray(initial_generals, dtype=np.int32)
    if len(ig) >= num_slots:
        return ig
    own = int(ig[perspective_slot])
    padded = np.full(num_slots, own, dtype=np.int32)
    padded[: len(ig)] = ig
    return padded


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ModelAgent:
    """One perspective. Owns its memory, sim dict, and an opaque reference
    to the (shared) model. Stateless across games — re-init via
    `init_for_game(...)` each new game.
    """

    def __init__(
        self,
        model: BCModel,
        perspective_slot: int,
        device: torch.device,
        force_move: bool = False,
        sample: bool = False,
        temperature: float = 1.0,
    ):
        """
        force_move: ignore the pass head when at least one move is legal
            under the mask. Useful when the BC pass head is biased toward
            "do nothing" (training data is dominated by pass frames) and
            we want to see the model actually play.
        sample: draw the move from softmax(masked policy logits) instead
            of taking the argmax. Adds variety; useful when argmax
            fixates on one cell. `temperature` scales the softmax.
        """
        self.model = model
        self.device = device
        self.perspective_slot = perspective_slot
        self.opp_slots = bc_obs.canonical_slot_order(perspective_slot, P)[1:]
        self.force_move = force_move
        self.sample = sample
        self.temperature = temperature

        # Filled by init_for_game
        self._H: int = 0
        self._W: int = 0
        self._ownership: list[np.ndarray] | None = None
        self._armies: list[np.ndarray] | None = None
        self._sim: dict[str, Any] | None = None
        self._memory: bc_obs.MemoryState | None = None
        self._bfs_cache: bc_bfs.BFSCache | None = None

        # Per-game diagnostics: counts of pass / sampled-or-argmax-move /
        # forced-skip (no legal moves available). Reset by init_for_game.
        self.n_passed = 0
        self.n_moved = 0
        self.n_no_legal = 0

        # Per-tick diagnostic series, reset by init_for_game. Captured every
        # tick the agent thinks: value/pass always; policy stats only when
        # the legality mask has at least one legal move. Pre-temperature
        # policy distribution — intrinsic model confidence, comparable across
        # temperature settings.
        self._value_exp_placement_per_tick: list[float] = []
        self._value_top_prob_per_tick: list[float] = []
        self._value_entropy_per_tick: list[float] = []
        self._pass_prob_per_tick: list[float] = []
        self._top1_prob_per_tick: list[float] = []
        self._top3_prob_per_tick: list[float] = []
        self._entropy_per_tick: list[float] = []

    def init_for_game(self, state: sim_core.State, static: Any) -> None:
        H, W = static.map_height, static.map_width

        self._H, self._W = H, W
        self._ownership = [np.asarray(state.snapshots_ownership[0], dtype=np.int8)]
        self._armies = [np.asarray(state.snapshots_armies[0], dtype=np.int16)]

        # initial_generals is length num_players in the static, but the BC
        # encoder was trained on 8-player FFA only (ELIGIBLE_PLAYER_COUNT=8)
        # and unconditionally indexes slots 0..P-1. For <8-player games we
        # pad the unused slots with the perspective's own general cell —
        # idempotent under fancy indexing (`general_cells_flat[ig] = True`
        # sets the own-general bit which is already set) and safe in the
        # `for p in range(P)` loop in step_memory (the `new_general` mask
        # excludes the own general because it's known from t=0, so the
        # padded slots never falsely claim a sighting).
        ig_padded = self._pad_initial_generals(static.initial_generals)

        self._sim = {
            "ownership": self._ownership,
            "armies": self._armies,
            "mountains": np.asarray(static.mountains, dtype=np.int32),
            "initial_cities": np.asarray(static.initial_cities, dtype=np.int32),
            "initial_generals": ig_padded,
            "cities": np.asarray(state.cities, dtype=np.int32),
            "cities_present_at": np.asarray(state.cities_present_at, dtype=np.int32),
            "capture_events": sim_adapter.capture_events_to_array(state.capture_events),
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

    def _pad_initial_generals(self, initial_generals) -> np.ndarray:
        return pad_initial_generals(initial_generals, self.perspective_slot)

    def act(self, state: sim_core.State, static: Any) -> tuple[int, int, int]:
        """Run inference for the current tick and return a wire move
        `(source, dest, is50)`. `(-1, -1, -1)` means "pass" — the driver
        loop should skip these (don't submit to sim_core.step_tick).
        """
        assert self._sim is not None, "call init_for_game first"
        assert self._memory is not None
        assert self._bfs_cache is not None
        assert self._ownership is not None
        assert self._armies is not None

        t = state.timestep
        H, W = self._H, self._W

        # 1. Append latest snapshot row
        own_t = np.asarray(state.snapshots_ownership[t], dtype=np.int8)
        arm_t = np.asarray(state.snapshots_armies[t], dtype=np.int16)
        self._ownership.append(own_t)
        self._armies.append(arm_t)

        # 2. Refresh dynamic sim fields. Cities can grow mid-game (general→
        # city on capture); capture_events grows whenever a capture fires.
        # The numpy arrays we built at init were snapshots; rebuild.
        self._sim["cities"] = np.asarray(state.cities, dtype=np.int32)
        self._sim["cities_present_at"] = np.asarray(
            state.cities_present_at, dtype=np.int32,
        )
        self._sim["capture_events"] = sim_adapter.capture_events_to_array(
            state.capture_events,
        )

        # 3. Append scoreboard row for tick t.
        land, army = bc_obs.scoreboard_row(own_t, arm_t, P)
        self._memory.land_count_history.append(land)
        self._memory.army_count_history.append(army)

        # 4. Visibility for this perspective
        vis = visibility.compute_visibility(own_t, self.perspective_slot, H, W)

        # 5. Advance memory + invalidate BFS cache if the known-passable
        # graph grew this tick.
        graph_grew = bc_obs.step_memory(
            self._memory, self._sim, t, vis, self.perspective_slot, H, W, P,
        )
        if graph_grew:
            self._bfs_cache.invalidate_graph()

        # 6. Build the obs tensor + legality mask
        obs_np = bc_obs.build_obs(
            self._sim, t, self.perspective_slot, self.opp_slots,
            vis, self._memory, self._bfs_cache, H, W,
        )
        mask_np = bc_mask.build_mask(self._sim, t, self.perspective_slot, H, W)

        # 7. Forward + select action
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)
        valid_mask = torch.zeros(1, 1, H_PADDED, W_PADDED, dtype=torch.bool, device=self.device)
        valid_mask[0, 0, :H, :W] = True
        with torch.no_grad():
            out = self.model(obs_tensor, valid_mask)
        masked_logits = flatten_policy_logits(out["policy_logits"], mask_tensor)

        has_legal = bool(mask_tensor.any())
        self._record_tick_diagnostics(out, masked_logits, has_legal)

        if not has_legal:
            # Degenerate case — no legal moves (e.g. very early game when
            # all owned tiles still have army<2). Pass is the only option.
            self.n_no_legal += 1
            return -1, -1, -1

        # Pass decision: sample from bernoulli(sigmoid(pass_logit)) under
        # `sample`, else hard threshold pass_logit > 0 (argmax-equivalent).
        # `force_move` overrides both.
        if not self.force_move:
            if self.sample:
                p_pass = torch.sigmoid(out["pass_logit"])
                pass_head_says_pass = bool(torch.bernoulli(p_pass).item())
            else:
                pass_head_says_pass = bool((out["pass_logit"] > 0).item())
            if pass_head_says_pass:
                self.n_passed += 1
                return -1, -1, -1

        if self.sample:
            # Temperature-scaled softmax sample over the legal action set.
            # masked_logits already has illegal slots set to MASK_NEG, so
            # softmax → ~0 probability on those.
            probs = torch.softmax(masked_logits / self.temperature, dim=1)
            flat_idx = int(torch.multinomial(probs, num_samples=1).item())
        else:
            flat_idx = int(masked_logits.argmax(dim=1).item())

        self.n_moved += 1
        return actions.decode(is_pass=False, flat_idx=flat_idx,
                              W_unpadded=W, W_padded=W_PADDED)

    def _record_tick_diagnostics(
        self,
        out: dict[str, torch.Tensor],
        masked_logits: torch.Tensor,
        has_legal: bool,
    ) -> None:
        # Value head: categorical placement over 8 classes (0=1st, 7=8th).
        # Pre-softmax logits → probabilities → derived stats.
        value_probs = torch.softmax(out["value_logits"][0], dim=0)
        placements = torch.arange(
            value_probs.size(0), dtype=value_probs.dtype, device=value_probs.device,
        )
        exp_placement = float((value_probs * placements).sum().item()) + 1.0  # 1..8 scale
        top_prob = float(value_probs.max().item())
        value_entropy = float(
            -(value_probs * (value_probs + 1e-12).log()).sum().item()
        )
        self._value_exp_placement_per_tick.append(exp_placement)
        self._value_top_prob_per_tick.append(top_prob)
        self._value_entropy_per_tick.append(value_entropy)

        # Pass head: sigmoid of scalar logit.
        self._pass_prob_per_tick.append(
            float(torch.sigmoid(out["pass_logit"][0]).item())
        )

        # Policy head: only meaningful when at least one legal slot exists.
        # Skip entirely when no legal moves — leaves these series sparser
        # than the value/pass series by n_no_legal entries.
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
        """Per-game diagnostic summary picked up by `MetricsCollector` via
        duck-typed `getattr(policy, "get_diagnostics", None)`.

        Lands in the eval-run output as `metrics["policy_diagnostics"][p]`
        in both `games/game_NNN.metrics.json` and the corresponding row of
        `results.jsonl`.
        """
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
