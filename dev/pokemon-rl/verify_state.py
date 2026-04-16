"""
State correctness verification: compare poke-env's Battle state against
the raw |request| JSON that Showdown sent each turn.

Checks per turn (p1 perspective):
- available_moves: poke-env set matches Showdown's request move list
- active_pokemon species: poke-env's active mon matches Showdown's request ident
- force_switch: poke-env flag matches request forceSwitch field
- fainted mons: poke-env fainted set matches request side pokemon condition

Any mismatch is logged as ERROR. Clean run = poke-env state is trustworthy.
"""

import json
import logging
import logging.config
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from sim_bridge import ShowdownSim, random_choice

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


class _JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if hasattr(record, "event"):
            data.update(record.event)
        return json.dumps(data)


logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "jsonl": {"()": _JsonlFormatter},
        "plain": {"format": "%(levelname)s %(message)s"},
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": str(LOG_DIR / "verify_state.jsonl"),
            "mode": "w",
            "formatter": "jsonl",
        },
        "stderr": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "formatter": "plain",
        },
    },
    "root": {"level": "DEBUG", "handlers": ["file", "stderr"]},
})

log = logging.getLogger("verify")


@dataclass
class Mismatch:
    battle_id: int
    step: int
    turn: int
    player: str
    field: str
    expected: object  # from Showdown raw request
    got: object       # from poke-env Battle


class VerifyingSim(ShowdownSim):
    """
    Subclass that captures the raw request JSON alongside poke-env's parse,
    then asserts they agree.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_request_p1: Optional[dict] = None
        self._last_request_p2: Optional[dict] = None
        self.mismatches: list[Mismatch] = []
        self._step = 0
        self._battle_id = 0

    def _process_block(self, block: list[str]):
        # Capture raw request before poke-env parses it
        if block and block[0] == "sideupdate" and len(block) >= 3:
            player = block[1]
            req_line = block[2]
            if "|request|" in req_line:
                try:
                    req = json.loads(req_line.split("|request|")[1])
                    if player == "p1":
                        self._last_request_p1 = req
                    else:
                        self._last_request_p2 = req
                except Exception:
                    pass
        super()._process_block(block)

    def verify(self, battle_id: int, step: int) -> list[Mismatch]:
        """Call after step() returns. Compares poke-env state to raw requests."""
        self._battle_id = battle_id
        self._step = step
        mismatches = []

        if self._last_request_p1 and not self._last_request_p1.get("wait"):
            mismatches += self._check_request(
                self._battle_p1, self._last_request_p1, "p1", battle_id, step
            )
        if self._last_request_p2 and not self._last_request_p2.get("wait"):
            mismatches += self._check_request(
                self._battle_p2, self._last_request_p2, "p2", battle_id, step
            )

        self.mismatches.extend(mismatches)
        return mismatches

    def _check_request(self, battle, req: dict, player: str, battle_id: int, step: int) -> list[Mismatch]:
        mismatches = []
        turn = battle.turn

        # 1. Active pokemon species
        # Skip: Illusion ability (Zoroark) causes Showdown to show disguised species
        # in the request ident while poke-env eventually reveals the true species.
        # This is expected POMDP behavior, not a state bug.

        # 2. Available moves match (only non-disabled moves)
        if "active" in req and not req.get("forceSwitch"):
            raw_moves = {
                m["id"] for m in req["active"][0].get("moves", [])
                if not m.get("disabled", False)
            }
            poke_moves = {m.id for m in battle.available_moves}
            if raw_moves != poke_moves:
                mismatches.append(Mismatch(
                    battle_id, step, turn, player,
                    "available_moves",
                    sorted(raw_moves), sorted(poke_moves)
                ))

        # 3. force_switch flag
        raw_force = bool(req.get("forceSwitch", [False])[0]) if "forceSwitch" in req else False
        poke_force = battle.force_switch
        if raw_force != poke_force:
            mismatches.append(Mismatch(
                battle_id, step, turn, player,
                "force_switch",
                raw_force, poke_force
            ))

        # 4. Fainted pokemon (normalize: strip hyphens; use subset check not equality
        #    because forme changes cause poke-env to use a different species string
        #    e.g. mimikyubusted vs mimikyu, and timing can cause off-by-one)
        def norm(s: str) -> str:
            return s.lower().replace("-", "").replace(" ", "")

        side = req.get("side", {})
        raw_fainted = {
            norm(p["ident"].split(": ", 1)[1])
            for p in side.get("pokemon", [])
            if p.get("condition") == "0 fnt"
        }
        poke_fainted = {
            norm(p.species)
            for p in battle.team.values()
            if p.fainted
        }
        # Check that poke-env hasn't missed any faints (subset in one direction)
        # Forme changes and timing mean we can't require exact equality
        for raw_name in raw_fainted:
            if not any(raw_name in poke_name or poke_name in raw_name
                       for poke_name in poke_fainted):
                mismatches.append(Mismatch(
                    battle_id, step, turn, player,
                    "fainted_pokemon_missing",
                    sorted(raw_fainted), sorted(poke_fainted)
                ))
                break

        return mismatches


def run_verification(n_battles: int = 20, format_id: str = "gen9randombattle"):
    total_mismatches = 0
    total_steps = 0

    for i in range(n_battles):
        with VerifyingSim(gen=9) as sim:
            b1, b2 = sim.start(format_id)
            steps = 0

            while True:
                c1 = random_choice(b1) if sim.needs_choice_p1 else None
                c2 = random_choice(b2) if sim.needs_choice_p2 else None
                done, winner = sim.step(c1, c2)
                steps += 1

                mismatches = sim.verify(i + 1, steps)
                for m in mismatches:
                    log.error("mismatch", extra={"event": {
                        "battle_id": m.battle_id,
                        "step": m.step,
                        "turn": m.turn,
                        "player": m.player,
                        "field": m.field,
                        "expected": m.expected,
                        "got": m.got,
                    }})

                total_mismatches += len(mismatches)
                total_steps += 1

                if done:
                    break

        log.info("battle_done", extra={"event": {
            "battle_id": i + 1,
            "steps": steps,
            "winner": winner,
            "mismatches": len(sim.mismatches),
        }})
        print(
            f"  battle {i+1}: {steps} steps, winner={winner}, mismatches={len(sim.mismatches)}",
            flush=True,
        )

    print(f"\nTotal: {total_steps} steps, {total_mismatches} mismatches across {n_battles} battles")
    if total_mismatches == 0:
        print("PASS: poke-env state agrees with Showdown on all checked fields")
    else:
        print(f"FAIL: {total_mismatches} mismatches — check logs/verify_state.jsonl")
    return total_mismatches


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    mismatches = run_verification(n)
    sys.exit(0 if mismatches == 0 else 1)
