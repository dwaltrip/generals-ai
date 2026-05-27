"""EvalBot — hand-coded greedy agent for evaluating NN training progress.

Same duck-typed interface as ModelAgent in self_play.agent:
  - init_for_game(state, static) — called once per game
  - act(view) — called once per tick with the current PlayerView
"""

from __future__ import annotations

from typing import Any

import sim_core

from eval_bot.attack import should_clear_attack, try_attack
from eval_bot.bot_config import BotConfig
from eval_bot.defend import should_clear_defend, try_defend
from eval_bot.expand import MOVE_EXPAND, MOVE_EXPLORE, pick_expand_or_explore
from eval_bot.killshot import try_killshot
from eval_bot.plan import AttackPlan, DefendPlan, GateResult, KillshotPlan, Plan, SingleMove
from eval_bot.world_model import PlayerView, WorldModel


def _is_spent(view: PlayerView, plan: Plan) -> bool:
    """Check if the plan can't make progress."""
    if plan.cursor >= len(plan.moves):
        return True

    src, dst, is50 = plan.moves[plan.cursor]
    src_r, src_c = divmod(src, view.W)
    dst_r, dst_c = divmod(dst, view.W)

    if int(view.own[src_r, src_c]) != view.my_slot:
        return True

    src_army = int(view.arm[src_r, src_c])
    if src_army <= 1:
        return True

    sent = src_army // 2 if is50 else src_army - 1

    dst_owner = int(view.own[dst_r, dst_c])
    if dst_owner != view.my_slot and dst_owner >= 0:
        dst_army = int(view.arm[dst_r, dst_c])
        if dst_army >= sent:
            return True

    return False


class EvalBot:
    def __init__(
        self,
        perspective_slot: int,
        cfg: BotConfig,
        max_ticks_hint: int = 2000,
    ):
        self.perspective_slot = perspective_slot
        self.cfg = cfg
        self.world = WorldModel(perspective_slot, self.cfg, max_ticks_hint)
        self._active_plan: Plan | None = None

        self.n_passed = 0
        self.n_moved = 0
        self.n_expand = 0
        self.n_explore = 0
        self.n_no_move = 0

    def init_for_game(self, state: sim_core.State, static: Any) -> None:
        self.world.init_for_game(state, static)
        self._active_plan = None

        self.n_passed = 0
        self.n_moved = 0
        self.n_expand = 0
        self.n_explore = 0
        self.n_no_move = 0

    def act(self, view: PlayerView) -> tuple[int, int, int]:
        """Decide one move for the current tick.

        Returns (source, dest, is50) or (-1, -1, -1) for pass.
        """
        move = self._tick_active_plan(view)
        if move is not None:
            self.n_moved += 1
            return move

        result = self._pick_new_plan(view)

        if result is None:
            self.n_no_move += 1
            return (-1, -1, -1)

        if isinstance(result, SingleMove):
            self.n_moved += 1
            if result.gate == MOVE_EXPAND:
                self.n_expand += 1
            elif result.gate == MOVE_EXPLORE:
                self.n_explore += 1
            return (result.src, result.dst, result.is50)

        # Plan variant — store and emit first move
        self._active_plan = result
        move = result.next_move()
        assert move is not None
        self.n_moved += 1
        return move

    def _tick_active_plan(
        self, view: PlayerView,
    ) -> tuple[int, int, int] | None:
        """Manage the active plan: re-evaluate, check preemption, emit.

        Returns a move if the plan produced one, None if cleared.
        Never creates new plans — that's _pick_new_plan's job.
        """
        if self._active_plan is None:
            return None

        plan = self._active_plan

        if _is_spent(view, plan):
            self._active_plan = None
            return None

        # gate-specific re-evaluation
        if isinstance(plan, DefendPlan):
            if should_clear_defend(view, plan):
                self._active_plan = None
                return None
        elif isinstance(plan, AttackPlan):
            if should_clear_attack(view, self.cfg, plan):
                self._active_plan = None
                return None

        # preemption: defend interrupts non-defend/killshot plans
        if not isinstance(plan, (DefendPlan, KillshotPlan)):
            if any(view.incursion_threats.values()):
                self._active_plan = None
                return None

        # emit
        move = plan.next_move()
        if move is None:
            self._active_plan = None
            return None

        return move

    def _pick_new_plan(self, view: PlayerView) -> GateResult:
        """Walk the decision ladder. First gate to fire wins."""
        defend_plan = try_defend(view, self.cfg)
        killshot_plan = try_killshot(view, self.cfg)

        if defend_plan and killshot_plan:
            # race condition: killshot wins if targeting the same player
            # and our gather arrives before the invader reaches our general
            if killshot_plan.target_player == defend_plan.target_player:
                tw = view.threat_windows[defend_plan.target_player][-1]
                my_travel = len(killshot_plan.moves)
                their_travel = tw.dist_to_general
                if my_travel + self.cfg.RACE_MARGIN < their_travel:
                    return killshot_plan
            return defend_plan
        if defend_plan:
            return defend_plan
        if killshot_plan:
            return killshot_plan

        # attack — spec puts expand/explore above attack, but
        # UnoccupiedFrontier > 0 is true nearly the entire game, so
        # expand/explore would almost always fire first and attack would
        # rarely trigger. Flipped so the bot presses weaker contacts
        # instead of passively expanding when an opportunity exists.
        attack_plan = try_attack(view, self.cfg)
        if attack_plan is not None:
            return attack_plan

        # expand/explore
        result = pick_expand_or_explore(view, self.cfg)
        if result is not None:
            return result

        return None
