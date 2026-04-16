"""
Smoke test + benchmark for sim_bridge.
Writes structured jsonl to logs/smoke.jsonl — tail -f it to watch live.

Run:
    mkdir -p logs
    PYTHONUNBUFFERED=1 uv run python smoke.py 2>&1 | tee logs/smoke_stderr.log
    # in another pane: tail -f logs/smoke.jsonl | python -m json.tool
"""

import json
import logging
import logging.config
import sys
import time
from pathlib import Path

from sim_bridge import ShowdownSim, random_choice, run_random_battle

# Subclass that logs every raw block for debugging
class DebugSim(ShowdownSim):
    def _process_block(self, block):
        if block:
            event = {
                "block_type": block[0],
                "lines": len(block),
                "preview": block[1][:120] if len(block) > 1 else "",
            }
            # For sideupdate blocks, extract player and request type
            if block[0] == "sideupdate" and len(block) >= 3:
                import json as _json
                event["player"] = block[1]
                event["raw_line"] = block[2][:200]
                try:
                    req = _json.loads(block[2].split("|request|")[1]) if "|request|" in block[2] else {}
                    event["is_wait"] = req.get("wait", False)
                    event["force_switch"] = bool(req.get("forceSwitch", [False])[0]) if "forceSwitch" in req else False
                    event["is_error"] = "|error|" in block[2]
                except Exception:
                    event["is_error"] = "|error|" in block[2]
            log.debug("raw_block", extra={"event": event})
        super()._process_block(block)

# ------------------------------------------------------------------
# Logging config: one jsonl handler to file, INFO+ to stderr
# ------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
JSONL_PATH = LOG_DIR / "smoke.jsonl"


class JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "event"):
            data.update(record.event)
        return json.dumps(data)


logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "jsonl": {"()": JsonlFormatter},
        "plain": {"format": "%(levelname)s %(name)s %(message)s"},
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": str(JSONL_PATH),
            "mode": "a",
            "formatter": "jsonl",
        },
        "stderr": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "formatter": "plain",
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["file", "stderr"],
    },
})

log = logging.getLogger("smoke")


# ------------------------------------------------------------------
# Instrumented battle runner
# ------------------------------------------------------------------

def run_instrumented_battle(battle_id: int, format_id: str) -> dict:
    t0 = time.perf_counter()
    log.info("battle_start", extra={"event": {"battle_id": battle_id, "format": format_id}})

    with DebugSim(gen=9) as sim:
        b1, b2 = sim.start(format_id)
        log.debug("battle_ready", extra={"event": {
            "battle_id": battle_id,
            "p1_active": b1.active_pokemon.species if b1.active_pokemon else None,
            "p2_active": b2.active_pokemon.species if b2.active_pokemon else None,
            "startup_ms": round((time.perf_counter() - t0) * 1000),
        }})

        steps = 0
        while True:
            t_step = time.perf_counter()
            c1 = random_choice(b1) if sim.needs_choice_p1 else None
            c2 = random_choice(b2) if sim.needs_choice_p2 else None
            done, winner = sim.step(c1, c2)
            step_ms = round((time.perf_counter() - t_step) * 1000, 1)
            steps += 1

            log.debug("step", extra={"event": {
                "battle_id": battle_id,
                "step": steps,
                "turn": b1.turn,
                "c1": c1,
                "c2": c2,
                "needs_p1": sim.needs_choice_p1,
                "needs_p2": sim.needs_choice_p2,
                "done": done,
                "step_ms": step_ms,
            }})

            if done:
                break

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    result = {
        "battle_id": battle_id,
        "format": format_id,
        "steps": steps,
        "winner": winner,
        "elapsed_ms": elapsed_ms,
    }
    log.info("battle_done", extra={"event": result})
    return result


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    FORMAT = "gen9randombattle"
    N = 10

    print(f"Running {N} battles in {FORMAT}", file=sys.stderr, flush=True)
    print(f"Logging to {JSONL_PATH}", file=sys.stderr, flush=True)
    print(f"Watch: tail -f {JSONL_PATH}", file=sys.stderr, flush=True)

    results = []
    for i in range(N):
        result = run_instrumented_battle(i + 1, FORMAT)
        print(
            f"  battle {i+1}: {result['steps']} steps, winner={result['winner']}, {result['elapsed_ms']}ms",
            file=sys.stderr,
            flush=True,
        )
        results.append(result)

    total_steps = sum(r["steps"] for r in results)
    total_ms = sum(r["elapsed_ms"] for r in results)
    print(f"\nResults ({N} battles):", file=sys.stderr, flush=True)
    print(f"  {N / (total_ms/1000):.1f} battles/sec", file=sys.stderr, flush=True)
    print(f"  {total_steps / (total_ms/1000):.0f} steps/sec", file=sys.stderr, flush=True)
    print(f"  {total_steps / N:.1f} avg steps/battle", file=sys.stderr, flush=True)
    print(f"  {total_ms / N:.0f} ms/battle avg", file=sys.stderr, flush=True)
