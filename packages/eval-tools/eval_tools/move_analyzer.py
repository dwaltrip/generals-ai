"""MoveAnalyzer — per-game action stream analysis.

Receives each tick's moves + state and accumulates stats. Designed to be
extensible: add new stats by adding a _record_* method and calling it
from record_moves, then exposing the result in summary().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import sim_core


@dataclass
class AttackChain:
    player: int
    length: int
    army_at_start: int


class MoveAnalyzer:
    def __init__(self, num_players: int):
        self.num_players = num_players

        # per-player counters
        self.ones_traversed: list[int] = [0] * num_players
        self.total_moves: list[int] = [0] * num_players
        self.total_passes: list[int] = [0] * num_players

        # attack chain tracking: per-player, the current chain being built
        # A chain is consecutive moves through enemy/neutral territory.
        self._current_chain: list[AttackChain | None] = [None] * num_players
        self._chain_gap: list[int] = [0] * num_players
        self.attack_chains: list[list[AttackChain]] = [[] for _ in range(num_players)]
        self.max_chain_gap: int = 2

        # per-player army sizes at attack moves (for percentile stats)
        self.attack_army_sizes: list[list[int]] = [[] for _ in range(num_players)]

    def record_moves(
        self,
        state: sim_core.State,
        moves: list[tuple[int, int, int, int]],
    ) -> None:
        own = np.asarray(state.ownership)
        arm = np.asarray(state.armies)

        players_that_moved = set()
        for player, src, dst, _is50 in moves:
            players_that_moved.add(player)
            self.total_moves[player] += 1

            # 1s-traversal: moving from an owned tile that now has army=1
            # (after the move executed, the source was left with 1)
            src_army_after = int(arm[src])
            if own[src] == player and src_army_after == 1:
                self.ones_traversed[player] += 1

            # attack detection: is the destination enemy or neutral?
            dst_owner_before_move = int(own[dst])
            is_attack_move = dst_owner_before_move != player and dst_owner_before_move != -1

            if is_attack_move:
                # src army before move isn't available post-step, but we can
                # approximate from the dst army if we captured, or track it
                # via the pre-step state. For now, record dst army as proxy
                # for attack size (it reflects the army that arrived).
                self.attack_army_sizes[player].append(int(arm[dst]))

            self._update_attack_chain(player, is_attack_move, int(arm[dst]))

        # players that were alive but didn't move this tick → passed
        for p in range(self.num_players):
            if state.alive[p] and p not in players_that_moved:
                self.total_passes[p] += 1
                self._tick_chain_gap(p)

    def _update_attack_chain(
        self, player: int, is_attack_move: bool, army_size: int,
    ) -> None:
        chain = self._current_chain[player]

        if is_attack_move:
            if chain is None:
                chain = AttackChain(
                    player=player, length=1, army_at_start=army_size,
                )
                self._current_chain[player] = chain
            else:
                chain.length += 1
            self._chain_gap[player] = 0
        else:
            self._tick_chain_gap(player)

    def _tick_chain_gap(self, player: int) -> None:
        chain = self._current_chain[player]
        if chain is None:
            return
        self._chain_gap[player] += 1
        if self._chain_gap[player] > self.max_chain_gap:
            self.attack_chains[player].append(chain)
            self._current_chain[player] = None
            self._chain_gap[player] = 0

    def finalize(self) -> None:
        for p in range(self.num_players):
            chain = self._current_chain[p]
            if chain is not None:
                self.attack_chains[p].append(chain)
                self._current_chain[p] = None

    def summary(self) -> dict:
        self.finalize()
        per_player = []
        for p in range(self.num_players):
            chains = self.attack_chains[p]
            chain_lengths = [c.length for c in chains]
            armies = self.attack_army_sizes[p]

            player_stats: dict = {
                "total_moves": self.total_moves[p],
                "total_passes": self.total_passes[p],
                "ones_traversed": self.ones_traversed[p],
                "attack_move_count": len(armies),
                "attack_chain_count": len(chains),
            }

            if chain_lengths:
                player_stats["attack_chain_avg_length"] = round(
                    sum(chain_lengths) / len(chain_lengths), 2,
                )
                player_stats["attack_chain_max_length"] = max(chain_lengths)
            if armies:
                sorted_armies = sorted(armies)
                n = len(sorted_armies)
                player_stats["attack_army_avg"] = round(sum(armies) / n, 1)
                player_stats["attack_army_p50"] = sorted_armies[n // 2]
                player_stats["attack_army_p90"] = sorted_armies[int(n * 0.9)]

            per_player.append(player_stats)

        return {"move_analysis": per_player}
