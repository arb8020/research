"""
Gymnasium environment wrapping ShowdownSim for single-player self-play.

Each reset() starts a new battle. The agent controls p1; p2 plays a fixed
opponent policy (default: random). step() returns p1's obs/reward/done.

Action space: Discrete(26) — matches SinglesEnv Gen 9 action mapping.
  0-5:   switch to team slot
  6-9:   move (no gimmick)
  10-13: move + mega evolve
  14-17: move + z-move
  18-21: move + dynamax
  22-25: move + terastallize

Observation space: Box(132,) float32 in [0, 1] — see obs.py.

Reward: +1 on win, -1 on loss, 0 each step.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

# Import only what we need from poke-env — avoid poke_env.environment which
# requires pettingzoo at import time.
from poke_env.battle.battle import Battle
from poke_env.battle.move import SPECIAL_MOVES
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    ForfeitBattleOrder,
    SingleBattleOrder,
)
from poke_env.player.player import Player

from sim_bridge import ShowdownSim, random_choice
from obs import embed_battle, obs_dim

# Gen 9: 6 switches + 4 moves × (base + mega + z + dynamax + tera) = 26
ACTION_SPACE_SIZE = 6 + 4 * 5  # = 26


def _action_to_order(action: int, battle: Battle) -> BattleOrder:
    """Inline of SinglesEnv.action_to_order for Gen 9, strict=False."""
    try:
        if action == -2:
            return DefaultBattleOrder()
        elif action == -1:
            return ForfeitBattleOrder()
        elif action < 6:
            return Player.create_order(list(battle.team.values())[action])
        else:
            avail_ids = [m.id for m in battle.available_moves]
            known_moves = list(battle.active_pokemon.moves.values())[:4]
            known_ids = [m.id for m in known_moves]
            mvs = (
                battle.available_moves
                if len(avail_ids) == 1 and avail_ids[0] not in known_ids
                else known_moves
            )
            idx = (action - 6) % 4
            return Player.create_order(
                mvs[idx],
                mega=10 <= action < 14,
                z_move=14 <= action < 18,
                dynamax=18 <= action < 22,
                terastallize=22 <= action < 26,
            )
    except Exception:
        return Player.choose_random_singles_move(battle)


def _get_action_mask(battle: Battle) -> np.ndarray:
    """Inline of SinglesEnv.get_action_mask for Gen 9."""
    switch_space = [
        i
        for i, pokemon in enumerate(battle.team.values())
        if not battle.trapped
        and pokemon.base_species in [p.base_species for p in battle.available_switches]
    ]
    if battle._wait:
        actions = [0]
    elif battle.active_pokemon is None:
        actions = switch_space
    else:
        known_moves = list(battle.active_pokemon.moves.values())[:4]
        avail_ids = [m.id for m in battle.available_moves]
        move_space = [
            i + 6
            for i, move in enumerate(known_moves)
            if move.id in avail_ids
        ]
        if not move_space and len(battle.available_moves) == 1 and battle.available_moves[0].id not in SPECIAL_MOVES:
            move_space = [6]
        available_z_ids = [m.id for m in battle.active_pokemon.available_z_moves]
        mega_space  = [i + 4  for i in move_space if battle.can_mega_evolve]
        zmove_space = [
            i + 6 + 8
            for i, move in enumerate(known_moves)
            if battle.can_z_move and move.id in avail_ids and move.id in available_z_ids
        ]
        dynamax_space = [i + 12 for i in move_space if battle.can_dynamax]
        tera_space    = [i + 16 for i in move_space if battle.can_tera]
        if not move_space and len(battle.available_moves) == 1 and battle.available_moves[0].id in SPECIAL_MOVES:
            move_space = [6]
        actions = switch_space + move_space + mega_space + zmove_space + dynamax_space + tera_space
    mask = [int(i in actions) for i in range(ACTION_SPACE_SIZE)]
    return np.array(mask, dtype=np.int8)


def _random_opponent(battle) -> str:
    """Default opponent: pure random legal move."""
    return random_choice(battle)


class PokemonEnv(gym.Env):
    """
    Single-agent Gymnasium env. Agent is p1; opponent policy is injected.

    Args:
        format_id: Showdown format string (default gen9randombattle).
        opponent_fn: Callable(battle) -> choice_str for p2. Defaults to random.
        p1_team: Packed team string for p1 (None = random/format default).
        p2_team: Packed team string for p2 (None = random/format default).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        format_id: str = "gen9randombattle",
        opponent_fn: Optional[Callable] = None,
        p1_team: Optional[str] = None,
        p2_team: Optional[str] = None,
    ):
        super().__init__()
        self.format_id = format_id
        self.opponent_fn = opponent_fn or _random_opponent
        self.p1_team = p1_team
        self.p2_team = p2_team

        self.observation_space = Box(
            low=0.0, high=1.0, shape=(obs_dim(),), dtype=np.float32
        )
        self.action_space = Discrete(ACTION_SPACE_SIZE)

        self._sim: Optional[ShowdownSim] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if self._sim is not None:
            self._sim.close()
            self._sim = None

        for attempt in range(5):
            sim = ShowdownSim(gen=9)
            self._b1, self._b2 = sim.start(
                self.format_id,
                p1_team=self.p1_team,
                p2_team=self.p2_team,
            )
            if not sim.done:
                break
            # Node process was defective (returned immediately) — discard and retry
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "env reset attempt %d: battle finished immediately (winner=%s), retrying",
                attempt + 1, sim.winner,
            )
            sim.close()
        else:
            raise RuntimeError(f"env reset failed after 5 attempts: Node process always exits immediately")
        self._sim = sim

        obs = embed_battle(self._b1)
        info = {"action_mask": self._action_mask()}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._sim is not None, "call reset() before step()"
        assert not self._sim.done

        c1 = self._action_to_choice(action) if self._sim.needs_choice_p1 else None
        c2 = self.opponent_fn(self._b2) if self._sim.needs_choice_p2 else None

        done, winner = self._sim.step(c1, c2)

        obs = embed_battle(self._b1)

        if done:
            reward = 1.0 if winner == "p1" else (-1.0 if winner == "p2" else 0.0)
        else:
            reward = 0.0

        info = {"action_mask": self._action_mask(), "winner": winner if done else None}
        return obs, reward, done, False, info

    def close(self):
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _action_mask(self) -> np.ndarray:
        if self._b1 is None:
            return np.ones(ACTION_SPACE_SIZE, dtype=np.int8)
        return _get_action_mask(self._b1)

    def _action_to_choice(self, action: int) -> str:
        order = _action_to_order(action, self._b1)
        return order.message.removeprefix("/choose ")
