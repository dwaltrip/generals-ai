"""BC model inference — turn a trained BCModel into a game-playing brain.

This is the model's *live inference contract*, kept next to the model and the
observation encoder it depends on. It consumes the neutral
`game_types.PlayerView` (raw per-tick game arrays) rather than `sim_core.State`,
so this module — and therefore `training/bc` — has no dependency on the game
engine. The `State -> PlayerView` translation lives in `game-runner`.

Two pieces, split at the GPU/CPU sync boundary so a vectorized runner can batch
forwards across many perspectives and pay a single sync per tick:

  - `BCModelHandle`: shared per checkpoint. Owns the model + device. `forward_batch`
    runs the batched forward; `decode_batch` does all the policy/pass/value math
    for the batch and pulls the per-row results to CPU in one sync.
  - `BCPerspective`: one (game, slot). Owns the growing sim dict, `MemoryState`,
    `BFSCache`, decode options, and per-tick diagnostics. `encode(view)` builds
    the obs tensor + masks (CPU numpy); `select_action(decision)` consumes a
    decoded row (already off-GPU) — sync-free.

Fog-of-war and the 8-slot general padding are applied here (in the encoder),
not by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bc import actions, visibility
from bc import bfs as bc_bfs
from bc import mask as bc_mask
from bc import obs as bc_obs
from training.bc.checkpoint import is_arch_bearing, load_bc_model
from training.bc.constants import H_PADDED, W_PADDED
from training.bc.loss import flatten_policy_logits
from training.bc.model import BCModel
from training.bc.obs import pad_initial_generals
from training.bc.obs_config import ObsConfig
from game_types import ObsBundle, PlayerView


# Fixed slot count the model + obs encoder were trained on (8-player FFA).
P = 8


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# One batched forward's raw output (batch dim intact). Opaque to the generic
# runner, which only shuttles it from forward_batch into decode_batch.
ModelOutput = dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class DecodeConfig:
    """Per-perspective decode options, gathered by the runner and handed to
    `decode_batch` so the batched math can branch per row. `generator` is the
    per-game RNG for reproducible sampling (None → global torch RNG); the runner
    seeds it per game.
    """

    force_move: bool
    sample: bool
    temperature: float
    generator: torch.Generator | None = None


@dataclass(frozen=True, slots=True)
class RowDecision:
    """`decode_batch` output for one row — plain CPU data, no GPU tensors. The
    GPU→CPU sync already happened; `select_action` just files it.
    """

    flat_idx: int                 # chosen action index, or -1 for pass/no-legal
    category: str                 # "moved" | "passed" | "no_legal"
    diag: dict[str, float]        # this tick's diagnostic scalars


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
        """Load a checkpoint into a handle. `value_head_variant` is a legacy-only
        fallback — used when the checkpoint has no `arch` key, ignored otherwise.

        The `model_key` (which a batched runner groups by) drops the variant for
        arch-bearing checkpoints — the arch is authoritative, so two loads under
        different fallback variants are the same model and should share a forward.
        Legacy checkpoints keep the variant: the same path loaded under different
        fallback variants is a different model, and the keys must not collide.
        """
        model = load_bc_model(path, device, value_head_variant)
        if is_arch_bearing(path, device):
            model_key = f"{path}|{device}"
        else:
            model_key = f"{path}|{device}|{value_head_variant}"
        return cls(model, device, model_key=model_key)

    def forward_batch(
        self, obs_batch: np.ndarray, valid_batch: np.ndarray,
    ) -> ModelOutput:
        """Run one forward over a stacked batch; return the raw batched output.

        obs_batch:   fp16/fp32 [B, OBS_CHANNELS, H_PADDED, W_PADDED]
        valid_batch: bool      [B, 1, H_PADDED, W_PADDED]
        Returns {policy_logits [B,8,H,W], pass_logit [B], value_logits [B,8]}.
        """
        # Inference runs fp32 (no autocast), so coerce obs to fp32 — a no-op for
        # an fp32 obs, an on-device upcast for an fp16 one (matches the training
        # consumer's `obs_for_model` no-autocast branch).
        obs_t = torch.from_numpy(obs_batch).to(self.device).float()
        valid_t = torch.from_numpy(valid_batch).to(self.device)
        with torch.no_grad():
            return self.model(obs_t, valid_t)

    def decode_batch(
        self, out: ModelOutput, mask_batch: np.ndarray, configs: list[DecodeConfig],
    ) -> list[RowDecision]:
        """Turn a batched forward output into one `RowDecision` per row, paying a
        single GPU→CPU sync. All policy/pass/value math is batched; per-row
        sampling (multinomial/bernoulli) draws stay on-GPU so the bulk pull is
        still one sync. The CPU half (category + bookkeeping) is `select_action`.

        mask_batch: bool [B, H, W, 8] per-cell legality (stacked policy masks).
        """
        device = self.device
        B = out["pass_logit"].shape[0]
        mask_t = torch.from_numpy(mask_batch).to(device)               # [B,H,W,8]
        masked = flatten_policy_logits(out["policy_logits"], mask_t)   # [B, H*W*8]
        has_legal = mask_t.reshape(B, -1).any(dim=1)                   # [B]

        # Value head: categorical placement over 8 classes (0=1st … 7=8th).
        value_probs = torch.softmax(out["value_logits"], dim=1)        # [B,8]
        placements = torch.arange(
            value_probs.shape[1], dtype=value_probs.dtype, device=device,
        )
        exp_placement = (value_probs * placements).sum(dim=1) + 1.0    # [B], 1..8
        value_top = value_probs.max(dim=1).values
        value_entropy = -(value_probs * (value_probs + 1e-12).log()).sum(dim=1)

        # Pass head.
        pass_prob = torch.sigmoid(out["pass_logit"])                   # [B]
        hard_pass = out["pass_logit"] > 0                              # [B] (argmax path)

        # Policy head diagnostics (untempered, masked).
        pol_probs = torch.softmax(masked, dim=1)                       # [B, H*W*8]
        top_k = torch.topk(pol_probs, k=min(3, pol_probs.shape[1]), dim=1).values
        policy_top1 = top_k[:, 0]
        policy_top3 = top_k.sum(dim=1)
        policy_entropy = -(pol_probs * (pol_probs + 1e-12).log()).sum(dim=1)

        # Move + pass decision: batched argmax / hard-threshold by default;
        # sampling rows overwrite their own slot (draws stay on-GPU).
        move_idx = masked.argmax(dim=1)                                # [B] long
        pass_says_pass = hard_pass.clone()
        for i, cfg in enumerate(configs):
            if cfg.sample:
                probs_i = torch.softmax(masked[i] / cfg.temperature, dim=0)
                move_idx[i] = torch.multinomial(probs_i, 1, generator=cfg.generator)[0]
                pass_says_pass[i] = (
                    torch.bernoulli(pass_prob[i], generator=cfg.generator) > 0.5
                )

        # ONE bulk sync: every per-row scalar pulled together.
        cols = torch.stack(
            [
                exp_placement, value_top, value_entropy, pass_prob,
                policy_top1, policy_top3, policy_entropy,
                move_idx.to(value_probs.dtype),
                has_legal.to(value_probs.dtype),
                pass_says_pass.to(value_probs.dtype),
            ],
            dim=1,
        ).cpu().numpy()                                                # [B, 10]

        decisions: list[RowDecision] = []
        for i, cfg in enumerate(configs):
            r = cols[i]
            legal = bool(r[8])
            diag: dict[str, float] = {
                "value_exp_placement": float(r[0]),
                "value_top_prob": float(r[1]),
                "value_entropy": float(r[2]),
                "pass_prob": float(r[3]),
            }
            if legal:
                diag["policy_top1"] = float(r[4])
                diag["policy_top3"] = float(r[5])
                diag["policy_entropy"] = float(r[6])

            if not legal:
                category, flat_idx = "no_legal", -1
            elif (not cfg.force_move) and bool(r[9]):
                category, flat_idx = "passed", -1
            else:
                category, flat_idx = "moved", int(r[7])
            decisions.append(RowDecision(flat_idx=flat_idx, category=category, diag=diag))
        return decisions


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
        obs_cfg: ObsConfig,
        *,
        force_move: bool = False,
        sample: bool = False,
        temperature: float = 1.0,
    ):
        self.perspective_slot = perspective_slot
        self.device = device
        # Obs-encoder config the model was trained with — source from the loaded
        # checkpoint's `handle.model.cfg.obs` so the built obs matches `in_ch`.
        self.obs_cfg = obs_cfg
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
        """Initialize per-game state from the t=0 view."""
        H, W = view.H, view.W
        self._H, self._W = H, W
        # encode() is the single owner of per-tick history — it appends one
        # snapshot + scoreboard row per tick, starting at tick 0 (the t=0 view
        # arrives again on encode's first call). Start empty; self._sim and
        # self._memory's history alias these lists, so encode's appends show up
        # through both.
        self._ownership = []
        self._armies = []

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
        self._memory = bc_obs.init_memory_common(
            self._sim, self.perspective_slot, H, W, self.obs_cfg, P,
        )
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

        Reads the neutral `PlayerView` (not `sim_core.State`): append the
        snapshot, refresh dynamic fields, step memory/BFS, build obs + masks.
        """
        assert self._sim is not None, "call reset() first"
        assert self._memory is not None
        assert self._bfs_cache is not None
        assert self._ownership is not None
        assert self._armies is not None

        t = view.timestep
        H, W = self._H, self._W

        own_t = np.asarray(view.ownership_t, dtype=np.int8)
        arm_t = np.asarray(view.armies_t, dtype=np.int16)

        # encode() is the single owner of per-tick history (reset() leaves it
        # empty): exactly one snapshot + scoreboard row per tick, appended in
        # order. Assert that contract so a runner sequencing bug fails loudly
        # here rather than silently feeding the model a tick-stale board — the
        # failure this single-owner contract exists to prevent.
        assert t == len(self._ownership), (
            f"encode() out of order: {len(self._ownership)} ticks recorded, got t={t}"
        )

        # 1. Append latest snapshot row.
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

        # 5. Advance memory. (BFS cache invalidation lives in `build_obs`.)
        bc_obs.step_memory(
            self._memory, self._sim, t, vis, self.perspective_slot, H, W, P,
        )

        # 6. Build obs + legality mask + valid-region mask.
        obs_np = bc_obs.build_obs(
            self._sim, t, self.perspective_slot, self.opp_slots,
            vis, self._memory, self._bfs_cache, H, W,
        )
        mask_np = bc_mask.build_mask(self._sim, t, self.perspective_slot, H, W)
        valid_np = np.zeros((1, H_PADDED, W_PADDED), dtype=bool)
        valid_np[0, :H, :W] = True

        return ObsBundle(obs=obs_np, policy_mask=mask_np, valid_mask=valid_np)

    @property
    def decode_config(self) -> DecodeConfig:
        """The per-row options `decode_batch` branches on for this perspective."""
        return DecodeConfig(
            force_move=self.force_move,
            sample=self.sample,
            temperature=self.temperature,
            generator=None,  # per-game seeding wired by the runner (fork #4)
        )

    def select_action(self, decision: RowDecision) -> tuple[int, int, int]:
        """Consume this perspective's decoded row (already off-GPU): record its
        diagnostics, bump the decision counter, and return the wire move.
        `(-1, -1, -1)` means pass / no legal move. Sync-free.
        """
        self._record_diag(decision.diag, has_legal=decision.category != "no_legal")

        if decision.category == "no_legal":
            self.n_no_legal += 1
            return -1, -1, -1
        if decision.category == "passed":
            self.n_passed += 1
            return -1, -1, -1

        self.n_moved += 1
        return actions.decode(
            is_pass=False, flat_idx=decision.flat_idx,
            W_unpadded=self._W, W_padded=W_PADDED,
        )

    def _record_diag(self, diag: dict[str, float], has_legal: bool) -> None:
        # Value + pass diagnostics are recorded every tick; policy diagnostics
        # only when at least one legal slot exists (matches the decoder's mask).
        self._value_exp_placement_per_tick.append(diag["value_exp_placement"])
        self._value_top_prob_per_tick.append(diag["value_top_prob"])
        self._value_entropy_per_tick.append(diag["value_entropy"])
        self._pass_prob_per_tick.append(diag["pass_prob"])
        if has_legal:
            self._top1_prob_per_tick.append(diag["policy_top1"])
            self._top3_prob_per_tick.append(diag["policy_top3"])
            self._entropy_per_tick.append(diag["policy_entropy"])

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
