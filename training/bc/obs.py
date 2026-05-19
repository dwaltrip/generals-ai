"""
Observation-tensor construction — the input to the policy/value network at
one (game, perspective, timestep) snapshot.

The spike uses a 12-channel placeholder (`[own_0..own_7, neutral, log_armies,
mountains, cities]`). The full network-architecture design expands this to
89 channels with fog-of-war computation, BFS distance maps, capture/contact
channels, and dense recent-history channels — deferred to post-spike work.

All channel assembly here treats the parser's ground-truth game state as
the perspective's view; no fog masking yet. When fog computation lands,
the channel set grows substantially but slot canonicalization stays put.
"""

from __future__ import annotations

import numpy as np

from bc.constants import H_PADDED, W_PADDED


def canonical_slot_order(perspective_slot: int, P: int = 8) -> list[int]:
    """
    Raw slot ids in canonical channel order. Channel 0 is always the perspective player.

    --- What this does ---

    The 8 ownership channels in the obs tensor get permuted so that channel 0
    holds the perspective player's cells, and channels 1..7 hold the opponents
    in ascending raw-slot order (with the perspective's own slot skipped).

    Example for P=8:
      perspective_slot=0 → [0, 1, 2, 3, 4, 5, 6, 7]
      perspective_slot=1 → [1, 0, 2, 3, 4, 5, 6, 7]
      perspective_slot=5 → [5, 0, 1, 2, 3, 4, 6, 7]

    --- Why bother (slot canonicalization, 101) ---

    Standard data-efficiency trick for multi-player game RL: by always placing
    the perspective player in a fixed channel, the network never has to "figure
    out who I am" from the input. That lookup is done deterministically at the
    dataloader stage, freeing model capacity to learn game dynamics instead.

    The K perspectives we extract per game then become K *augmented* views of
    the same game state — same dynamics, different "you" — rather than K
    decorrelated inputs. Without canonicalization the model would see those
    K views as 8 disconnected slot-specific input distributions, losing
    roughly a factor of K in data efficiency.

    --- Why ascending-skip ordering for the opponents ---

    Two patterns appear in the literature:
      • Ascending raw slot id, perspective removed ("ascending-skip"): used in
        Dota/OpenAI Five, StarCraft/AlphaStar, Pluribus (6p poker), and other
        non-cyclic-turn games.
      • Cyclic shift by turn order: used in Hanabi, Mahjong/Suphx, anywhere
        with a circular turn structure where "next player" has semantic meaning.

    Generals.io is simultaneous-action — no turn cycle — so the cyclic
    interpretation has no game-mechanics anchor. Ascending-skip matches the
    non-cyclic literature pattern and the spec phrasing ("opponents stay in
    their natural generals.io slot ordering").

    --- Known limitation (and the deferred fix) ---

    Canonicalization makes the model perspective-invariant within and across
    games, but it does NOT make opponent *identity* perspective-invariant: the
    same opponent (say, raw slot 3) lands in different canonical channels
    depending on which perspective is currently in slot 0. The real fix is
    slot-permutation augmentation, which randomizes opponent ordering across
    canonical channels 1..7 during training. Deferred for the spike; locked
    decision in the design docs.
    """
    return [perspective_slot] + [s for s in range(P) if s != perspective_slot]


def build_obs(
    sim: dict[str, np.ndarray],
    t: int,
    perspective_slot: int,
    H: int,
    W: int,
) -> np.ndarray:
    """
    Build the 12-channel obs tensor for one (perspective, timestep).

    Returns a `float32 [12, H_PADDED, W_PADDED]` array. The unpadded board
    occupies `[:, :H, :W]`; the right/bottom margins are zero-filled (top-left
    padding convention, see `bc/constants.py`).

    Channel layout (post-canonicalization):
      0..7  — ownership one-hot per canonicalized slot (channel 0 = perspective)
      8     — neutral mask (cells where `ownership == -1`)
      9     — log-armies, `log(1 + armies)`
      10    — mountains mask (static)
      11    — cities mask at time `t` (dynamic — see `cities_present_at`)
    """
    HW = H * W
    own_t = sim["ownership"][t]      # int8  [HW]; sentinels: -2 mountain, -1 neutral, 0..7 player
    armies_t = sim["armies"][t]      # int16 [HW]

    obs_unpadded = np.zeros((12, H, W), dtype=np.float32)

    # Channels 0..7 — ownership one-hots, permuted into canonical slot order.
    # See `canonical_slot_order` for the full design rationale.
    slot_order = canonical_slot_order(perspective_slot)
    for channel_idx, raw_slot in enumerate(slot_order):
        obs_unpadded[channel_idx] = (own_t == raw_slot).reshape(H, W)

    # Channel 8 — neutral mask. Distinct from the mountains channel: neutral
    # cells (sentinel -1) include neutral cities and empty plains, both of which
    # are capturable; mountains (sentinel -2) are not.
    obs_unpadded[8] = (own_t == -1).reshape(H, W)

    # Channel 9 — log-armies. Army counts are heavy-tailed (typically 0–10 in
    # the early game, hundreds late). `log(1+x)` compresses the tail into a
    # roughly linear-friendly range while keeping 0 at 0. Standard scaling for
    # army-like quantities in board-game RL.
    obs_unpadded[9] = np.log1p(armies_t).reshape(H, W)

    # Channel 10 — mountains (static within a game). Parser stores mountains as
    # a sparse `[N]` index list; densify to a `[HW]` bool mask here. Per-frame
    # cost is microseconds; could lift to a per-game cache in the dataset's
    # outer loop if profiling shows it matters.
    mountains_mask = np.zeros(HW, dtype=bool)
    mountains_mask[sim["mountains"]] = True
    obs_unpadded[10] = mountains_mask.reshape(H, W)

    # Channel 11 — cities present at time `t`. Cities never disappear once they
    # exist; `cities_present_at[i]` is the snapshot at which `cities[i]` came
    # into being (initial cities have `cities_present_at == 0`; general-to-city
    # transitions on capture appear here too).
    cities_at_t = sim["cities"][sim["cities_present_at"] <= t]
    cities_mask = np.zeros(HW, dtype=bool)
    cities_mask[cities_at_t] = True
    obs_unpadded[11] = cities_mask.reshape(H, W)

    # Top-left pad to [12, H_PADDED, W_PADDED]. Zero fill is safe everywhere:
    # binary masks → False outside the board; log-armies → log(1+0) = 0. The
    # action mask (built in dataset.build_mask) excludes padded cells from all
    # legality, so the fill value is mostly cosmetic.
    obs = np.zeros((12, H_PADDED, W_PADDED), dtype=np.float32)
    obs[:, :H, :W] = obs_unpadded
    return obs
