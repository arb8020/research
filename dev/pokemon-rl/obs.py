"""
Battle observation encoder.

Returns a flat float32 numpy array each turn. Structure:

  [last_move_mine] [last_move_opp] [active_mine] [active_opp]
  [team_mine x6]  [team_opp x6]   [field]        [available_moves x4]

Categorical features are encoded as (value / max_value) so everything
stays in [0, 1]. Unknown/None maps to 0.

Call obs_dim() to get the vector length.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from poke_env.battle.battle import Battle
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.move_category import MoveCategory
from poke_env.data import GenData

# ---------------------------------------------------------------------------
# Vocab sizes (from survey)
# ---------------------------------------------------------------------------

_N_SPECIES = 1517   # 1516 gen9 entries + 1 unknown slot
_N_MOVES   = 955    # 954 gen9 moves + 1 unknown slot
_N_WEATHER = len(Weather)
_N_FIELD   = len(Field)
_N_SIDE    = len(SideCondition)
_N_STATUS  = len(Status)
_N_TYPE    = len(PokemonType)
_N_STATS   = 6      # hp atk def spa spd spe

_GEN_DATA  = GenData.from_gen(9)

# Build species and move string→int maps once at import time
_SPECIES_MAP: dict[str, int] = {
    name: idx + 1
    for idx, name in enumerate(sorted(_GEN_DATA.pokedex.keys()))
}
_MOVE_MAP: dict[str, int] = {
    name: idx + 1
    for idx, name in enumerate(sorted(_GEN_DATA.moves.keys()))
}


def _species_id(name: Optional[str]) -> float:
    if name is None:
        return 0.0
    return _SPECIES_MAP.get(name.lower(), 0) / _N_SPECIES


def _move_id(move_id: Optional[str]) -> float:
    if move_id is None:
        return 0.0
    return _MOVE_MAP.get(move_id.lower(), 0) / _N_MOVES


def _type_id(t: Optional[PokemonType]) -> float:
    if t is None:
        return 0.0
    return t.value / _N_TYPE


def _status_id(s: Optional[Status]) -> float:
    if s is None:
        return 0.0
    return s.value / _N_STATUS


def _category_id(c: Optional[MoveCategory]) -> float:
    if c is None:
        return 0.0
    return c.value / 3.0


# ---------------------------------------------------------------------------
# Sub-encoders
# ---------------------------------------------------------------------------

# Dimensions (update _DIMS if you change any encoder)
_DIM_LAST_MOVE = 6   # move_id, type, category, base_power/250, pp_frac, effectiveness
_DIM_ACTIVE    = 15  # species, type1, type2, hp_frac, status, boosts×6, is_tera, tera_type
_DIM_TEAM_SLOT = 4   # species, hp_frac, status, fainted
_DIM_FIELD     = (
    1             # weather
    + 1           # trick_room flag (most impactful terrain)
    + 4           # terrain (electric/grassy/misty/psychic)
    + 5           # my side: stealth_rock, spikes(0-3), toxic_spikes(0-2), reflect, light_screen
    + 5           # opp side: same
    + 2           # tailwind each side
)
_DIM_AVAIL_MOVE = 6  # same as last_move but for each available move


def _encode_last_move(
    mon: Optional[Pokemon],
    opponent: Optional[Pokemon],
    buf: np.ndarray,
    offset: int,
) -> int:
    """Encode the last move used by mon, with effectiveness vs opponent."""
    move: Optional[Move] = mon.last_move if mon is not None else None
    if move is not None:
        eff = (
            opponent.damage_multiplier(move)
            if opponent is not None else 1.0
        )
        buf[offset + 0] = _move_id(move.id)
        buf[offset + 1] = _type_id(move.type)
        buf[offset + 2] = _category_id(move.category)
        buf[offset + 3] = min(move.base_power / 250.0, 1.0)
        buf[offset + 4] = move.current_pp / max(move.max_pp, 1)
        buf[offset + 5] = eff / 4.0  # max effectiveness is 4×
    return offset + _DIM_LAST_MOVE


def _encode_active(
    mon: Optional[Pokemon],
    buf: np.ndarray,
    offset: int,
) -> int:
    if mon is not None:
        boosts = mon.boosts  # dict stat→int in [-6, 6]
        buf[offset + 0]  = _species_id(mon.species)
        buf[offset + 1]  = _type_id(mon.type_1)
        buf[offset + 2]  = _type_id(mon.type_2)
        buf[offset + 3]  = mon.current_hp_fraction
        buf[offset + 4]  = _status_id(mon.status)
        buf[offset + 5]  = (boosts.get("atk", 0) + 6) / 12.0
        buf[offset + 6]  = (boosts.get("def", 0) + 6) / 12.0
        buf[offset + 7]  = (boosts.get("spa", 0) + 6) / 12.0
        buf[offset + 8]  = (boosts.get("spd", 0) + 6) / 12.0
        buf[offset + 9]  = (boosts.get("spe", 0) + 6) / 12.0
        buf[offset + 10] = (boosts.get("evasion", 0) + 6) / 12.0
        buf[offset + 11] = (boosts.get("accuracy", 0) + 6) / 12.0
        buf[offset + 12] = 1.0 if mon.is_terastallized else 0.0
        buf[offset + 13] = _type_id(mon.tera_type)
        buf[offset + 14] = 1.0 if mon.protect_counter > 0 else 0.0
    return offset + _DIM_ACTIVE


def _encode_team_slot(
    mon: Optional[Pokemon],
    buf: np.ndarray,
    offset: int,
) -> int:
    if mon is not None:
        buf[offset + 0] = _species_id(mon.species)
        buf[offset + 1] = mon.current_hp_fraction
        buf[offset + 2] = _status_id(mon.status)
        buf[offset + 3] = 1.0 if mon.fainted else 0.0
    return offset + _DIM_TEAM_SLOT


def _encode_field(battle: Battle, buf: np.ndarray, offset: int) -> int:
    # Weather
    weather = next(iter(battle.weather), None) if battle.weather else None
    buf[offset] = weather.value / _N_WEATHER if weather else 0.0
    offset += 1

    # Trick room
    buf[offset] = 1.0 if Field.TRICK_ROOM in battle.fields else 0.0
    offset += 1

    # Terrain flags
    for terrain in (Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN,
                    Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN):
        buf[offset] = 1.0 if terrain in battle.fields else 0.0
        offset += 1

    # Side conditions — my side then opponent side
    for sc_dict in (battle.side_conditions, battle.opponent_side_conditions):
        buf[offset + 0] = 1.0 if SideCondition.STEALTH_ROCK in sc_dict else 0.0
        buf[offset + 1] = sc_dict.get(SideCondition.SPIKES, 0) / 3.0
        buf[offset + 2] = sc_dict.get(SideCondition.TOXIC_SPIKES, 0) / 2.0
        buf[offset + 3] = 1.0 if SideCondition.REFLECT in sc_dict else 0.0
        buf[offset + 4] = 1.0 if SideCondition.LIGHT_SCREEN in sc_dict else 0.0
        offset += 5

    # Tailwind each side
    buf[offset + 0] = 1.0 if SideCondition.TAILWIND in battle.side_conditions else 0.0
    buf[offset + 1] = 1.0 if SideCondition.TAILWIND in battle.opponent_side_conditions else 0.0
    offset += 2

    return offset


def _encode_available_move(
    move: Optional[Move],
    opponent: Optional[Pokemon],
    buf: np.ndarray,
    offset: int,
) -> int:
    if move is not None:
        eff = opponent.damage_multiplier(move) if opponent is not None else 1.0
        buf[offset + 0] = _move_id(move.id)
        buf[offset + 1] = _type_id(move.type)
        buf[offset + 2] = _category_id(move.category)
        buf[offset + 3] = min(move.base_power / 250.0, 1.0)
        buf[offset + 4] = move.current_pp / max(move.max_pp, 1)
        buf[offset + 5] = eff / 4.0
    return offset + _DIM_AVAIL_MOVE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_OBS_DIM = (
    _DIM_LAST_MOVE * 2        # last move: mine + opponent
    + _DIM_ACTIVE * 2         # active mon: mine + opponent
    + _DIM_TEAM_SLOT * 6      # my team (6 slots)
    + _DIM_TEAM_SLOT * 6      # opponent team (6 slots)
    + _DIM_FIELD              # field conditions
    + _DIM_AVAIL_MOVE * 4     # available moves (4 slots)
)


def obs_dim() -> int:
    return _OBS_DIM


def embed_battle(battle: Battle) -> np.ndarray:
    """
    Encode the current battle state into a flat float32 observation vector.
    Unknown/unrevealed values are encoded as 0.
    """
    buf = np.zeros(_OBS_DIM, dtype=np.float32)
    offset = 0

    my_active  = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon

    # Last moves used this turn
    offset = _encode_last_move(my_active, opp_active, buf, offset)
    offset = _encode_last_move(opp_active, my_active, buf, offset)

    # Active mons
    offset = _encode_active(my_active, buf, offset)
    offset = _encode_active(opp_active, buf, offset)

    # Teams — pad to 6 slots with zeros for unrevealed/missing
    my_team  = list(battle.team.values())
    opp_team = list(battle.opponent_team.values())
    for i in range(6):
        offset = _encode_team_slot(my_team[i]  if i < len(my_team)  else None, buf, offset)
    for i in range(6):
        offset = _encode_team_slot(opp_team[i] if i < len(opp_team) else None, buf, offset)

    # Field
    offset = _encode_field(battle, buf, offset)

    # Available moves (up to 4)
    avail = battle.available_moves
    for i in range(4):
        offset = _encode_available_move(
            avail[i] if i < len(avail) else None,
            opp_active,
            buf,
            offset,
        )

    assert offset == _OBS_DIM, f"obs encoding bug: wrote {offset}, expected {_OBS_DIM}"
    return buf
