"""
Subprocess bridge to Pokemon Showdown's simulate-battle command.

Design: fully synchronous battle I/O, intended to be run in a thread
or process pool by the training loop. No Trio internals involved.

Usage (synchronous):
    sim = ShowdownSim(gen=9)
    b1, b2 = sim.start("gen9randombattle")
    done, winner = sim.step("move 1", "move 2")
    done, winner = sim.step(None, "switch 3")  # force-switch
    sim.close()

Usage (async via thread):
    async def run_battle():
        return await trio.to_thread.run_sync(run_one_battle, cancellable=True)
"""

import json
import logging
import os
import queue
import random
import subprocess
import threading
from pathlib import Path
from typing import Optional

from poke_env.battle.battle import Battle
from poke_env.battle.double_battle import DoubleBattle


NODE_BIN = os.environ.get("NODE_BIN", "/Users/chiraagbalu/.nvm/versions/node/v20.20.2/bin/node")
SHOWDOWN_PATH = Path(os.environ.get("SHOWDOWN_PATH", "pokemon-showdown/pokemon-showdown"))

logger = logging.getLogger(__name__)


def _spawn_node_proc() -> subprocess.Popen:
    return subprocess.Popen(
        [NODE_BIN, str(SHOWDOWN_PATH.resolve()), "simulate-battle", "--skip-build"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class NodeProcessPool:
    """
    Pre-warmed pool of idle Node simulate-battle processes.

    Claim a process with acquire() — returns immediately if one is ready,
    blocks until one is available otherwise. After a battle ends, call
    release_and_refill() to kill the used process and background-spawn a
    replacement so the pool stays full.

    Thread-safe. Initialize once per training run via init().
    """

    def __init__(self, size: int):
        self._size = size
        self._q: queue.Queue[subprocess.Popen] = queue.Queue()
        self._lock = threading.Lock()
        self._filling = 0  # background spawns in flight

        logger.info("NodeProcessPool: pre-warming %d Node processes (warmup=%.0fs)", size, self._WARMUP_SECS)
        import time as _time
        for i in range(size):
            self._spawn_into_pool()
            # Stagger spawns by 0.5s to avoid simultaneous dist file reads
            # that can corrupt Showdown's JS module loading on cold containers.
            if i < size - 1:
                _time.sleep(0.5)

    # Seconds to wait after Popen before enqueuing — gives Node time to load
    # Showdown (~1s locally, ~15s on Modal cold container).
    _WARMUP_SECS: float = 0.0  # set by init_pool

    def _spawn_into_pool(self):
        """Spawn one process in a background thread and enqueue it when ready."""
        with self._lock:
            self._filling += 1

        def _do_spawn():
            try:
                proc = _spawn_node_proc()
                if self._WARMUP_SECS > 0:
                    import time as _time
                    _time.sleep(self._WARMUP_SECS)
                self._q.put(proc)
                logger.debug("NodeProcessPool: process ready (pool size ~%d)", self._q.qsize())
            except Exception:
                logger.exception("NodeProcessPool: spawn failed")
            finally:
                with self._lock:
                    self._filling -= 1

        threading.Thread(target=_do_spawn, daemon=True).start()

    def acquire(self, timeout: float = 120.0) -> subprocess.Popen:
        """Claim a ready process. Blocks up to timeout seconds."""
        try:
            proc = self._q.get(timeout=timeout)
            logger.debug("NodeProcessPool: acquired process (pool size ~%d)", self._q.qsize())
            return proc
        except queue.Empty:
            raise RuntimeError(
                f"NodeProcessPool: no process available after {timeout}s "
                f"(pool_size={self._size}, filling={self._filling})"
            )

    def release_and_refill(self, proc: subprocess.Popen):
        """Kill the used process and spawn a replacement in the background."""
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
        self._spawn_into_pool()

    def close(self):
        """Drain and kill all pooled processes."""
        while True:
            try:
                proc = self._q.get_nowait()
                try:
                    proc.terminate()
                except Exception:
                    pass
            except queue.Empty:
                break


# Module-level pool — initialized on first ShowdownSim.start() if pool_size > 0.
_pool: Optional[NodeProcessPool] = None
_pool_lock = threading.Lock()


def init_pool(size: int, warmup_secs: float = 0.0):
    """Call once before training to pre-warm Node processes.

    warmup_secs: seconds to wait after Popen before considering the process
    ready. Set to ~15-20 on Modal where Node startup takes ~12-15s.
    """
    global _pool
    with _pool_lock:
        if _pool is None:
            pool = NodeProcessPool(size)
            pool._WARMUP_SECS = warmup_secs
            _pool = pool


def close_pool():
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


class ShowdownSim:
    """
    Fully synchronous wrapper around `pokemon-showdown simulate-battle`.
    One battle per instance. Not thread-safe; create one per thread.
    """

    def __init__(
        self,
        showdown_path: Path = SHOWDOWN_PATH,
        gen: int = 9,
        doubles: bool = False,
    ):
        self._showdown_path = Path(showdown_path).resolve()
        self._gen = gen
        self._doubles = doubles
        self._proc: Optional[subprocess.Popen] = None
        self._battle_p1: Optional[Battle] = None
        self._battle_p2: Optional[Battle] = None
        self._done: bool = False
        self._winner: Optional[str] = None
        self._needs_choice_p1: bool = False
        self._needs_choice_p2: bool = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def start(
        self,
        format_id: str,
        p1_team: Optional[str] = None,
        p2_team: Optional[str] = None,
    ):
        """Start a battle. Returns (battle_p1, battle_p2) once both sides have requests."""
        assert self._proc is None

        if _pool is not None:
            self._proc = _pool.acquire()
        else:
            self._proc = _spawn_node_proc()

        BattleClass = DoubleBattle if self._doubles else Battle
        self._battle_p1 = BattleClass(
            battle_tag=f"battle-{format_id}-1",
            username="p1",
            logger=logging.getLogger("sim.p1"),
            gen=self._gen,
        )
        self._battle_p2 = BattleClass(
            battle_tag=f"battle-{format_id}-1",
            username="p2",
            logger=logging.getLogger("sim.p2"),
            gen=self._gen,
        )
        self._done = False
        self._winner = None
        self._needs_choice_p1 = False
        self._needs_choice_p2 = False

        p1_spec: dict = {"name": "p1"}
        p2_spec: dict = {"name": "p2"}
        if p1_team:
            p1_spec["team"] = p1_team
        if p2_team:
            p2_spec["team"] = p2_team

        self._write(f'>start {{"formatid":"{format_id}"}}')
        self._write(f">player p1 {json.dumps(p1_spec)}")
        self._write(f">player p2 {json.dumps(p2_spec)}")

        self._read_until_choices_needed()
        return self._battle_p1, self._battle_p2

    def step(self, choice_p1: Optional[str], choice_p2: Optional[str]):
        """Send choices and advance. Returns (done, winner)."""
        assert not self._done
        assert self._needs_choice_p1 == (choice_p1 is not None), (
            f"p1: needs_choice={self._needs_choice_p1} choice={choice_p1!r}"
        )
        assert self._needs_choice_p2 == (choice_p2 is not None), (
            f"p2: needs_choice={self._needs_choice_p2} choice={choice_p2!r}"
        )

        if choice_p1 is not None:
            self._needs_choice_p1 = False
            self._write(f">p1 {choice_p1}")
        if choice_p2 is not None:
            self._needs_choice_p2 = False
            self._write(f">p2 {choice_p2}")

        self._read_until_choices_needed()
        return self._done, self._winner

    def close(self):
        if self._proc is not None:
            proc = self._proc
            self._proc = None
            if _pool is not None:
                _pool.release_and_refill(proc)
            else:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                except Exception:
                    pass

    @property
    def needs_choice_p1(self) -> bool:
        return self._needs_choice_p1

    @property
    def needs_choice_p2(self) -> bool:
        return self._needs_choice_p2

    @property
    def done(self) -> bool:
        return self._done

    @property
    def winner(self) -> Optional[str]:
        return self._winner

    # ------------------------------------------------------------------
    # Internal — all synchronous blocking I/O
    # ------------------------------------------------------------------

    def _write(self, line: str):
        self._proc.stdin.write((line + "\n").encode())
        self._proc.stdin.flush()

    def _readline(self) -> Optional[str]:
        raw = self._proc.stdout.readline()
        if not raw:
            return None
        return raw.rstrip(b"\n").decode(errors="replace")

    def _read_until_choices_needed(self):
        """
        Read blocks until we have all the requests for this turn.

        Showdown sends one sideupdate per player per turn:
        - Normal turn: p1 choice request + p2 choice request
        - Force-switch turn: one wait request + one forceSwitch request

        We return when we've received all sideupdates for this turn,
        determined by: both choice flags set, or one choice + one explicit wait THIS cycle.
        """
        current_block: list[str] = []
        got_wait_p1 = False  # received wait=true for p1 THIS cycle
        got_wait_p2 = False  # received wait=true for p2 THIS cycle

        while True:
            line = self._readline()
            if line is None:
                if current_block:
                    self._process_block(current_block)
                self._done = True
                # Log stderr so we can see why Node exited
                try:
                    stderr_out = self._proc.stderr.read().decode(errors="replace").strip()
                    if stderr_out:
                        logger.error("showdown stderr: %s", stderr_out[:2000])
                except Exception:
                    pass
                return

            if line == "":
                # Peek at block type before processing to track wait flags
                block_type = current_block[0] if current_block else None
                if block_type == "sideupdate" and len(current_block) >= 3:
                    player = current_block[1]
                    req_line = current_block[2]
                    if "|request|" in req_line:
                        try:
                            req = json.loads(req_line.split("|request|")[1])
                            if req.get("wait", False):
                                if player == "p1":
                                    got_wait_p1 = True
                                else:
                                    got_wait_p2 = True
                        except Exception:
                            pass

                self._process_block(current_block)
                current_block = []

                if self._done:
                    return
                # Both need choices (normal turn)
                if self._needs_choice_p1 and self._needs_choice_p2:
                    return
                # Force-switch: one needs choice, other explicitly waited THIS cycle
                if self._needs_choice_p1 and got_wait_p2:
                    return
                if self._needs_choice_p2 and got_wait_p1:
                    return
            else:
                current_block.append(line)

    def _process_block(self, block: list[str]):
        if not block:
            return

        block_type = block[0]

        if block_type == "update":
            for line in block[1:]:
                if not line or line == "|":
                    continue
                split = line.split("|")
                if len(split) < 2:
                    continue
                msg_type = split[1]
                if msg_type == "win":
                    self._done = True
                    self._winner = split[2].strip() if len(split) > 2 else None
                elif msg_type == "tie":
                    self._done = True
                    self._winner = None
                else:
                    try:
                        self._battle_p1.parse_message(split)
                        self._battle_p2.parse_message(split)
                    except NotImplementedError:
                        pass

        elif block_type == "sideupdate":
            if len(block) < 3:
                return
            player = block[1]
            msg_line = block[2]
            split = msg_line.split("|")
            if len(split) >= 3 and split[1] == "request":
                request = json.loads(split[2])
                is_wait = request.get("wait", False)
                if player == "p1":
                    self._battle_p1.parse_request(request)
                    if not is_wait:
                        self._needs_choice_p1 = True
                else:
                    self._battle_p2.parse_request(request)
                    if not is_wait:
                        self._needs_choice_p2 = True

        elif block_type == "end":
            self._done = True


def random_choice(battle) -> str:
    """
    Pick a random legal choice using poke-env's RandomPlayer logic.
    Returns the wire-format string to send after >p1 / >p2 (no /choose prefix).

    poke-env generates /choose move {id} or /choose switch {name}.
    We strip the /choose prefix since the >pN prefix serves that role.
    """
    from poke_env.player.baselines import RandomPlayer
    order = RandomPlayer.choose_random_move(battle)
    # order.message returns "/choose move thunderbolt" or "/choose switch pikachu"
    # strip the "/choose " prefix for use in the simulate-battle protocol
    return order.message.removeprefix("/choose ")


def run_random_battle(format_id: str = "gen9randombattle") -> dict:
    """Run one complete battle with random choices. Returns result dict."""
    with ShowdownSim(gen=9) as sim:
        b1, b2 = sim.start(format_id)
        steps = 0
        while True:
            c1 = random_choice(b1) if sim.needs_choice_p1 else None
            c2 = random_choice(b2) if sim.needs_choice_p2 else None
            done, winner = sim.step(c1, c2)
            steps += 1
            if done:
                return {"steps": steps, "winner": winner}


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import time
    import logging as _logging
    _logging.basicConfig(level=_logging.WARNING)

    print("Single battle smoke test...")
    with ShowdownSim(gen=9) as sim:
        b1, b2 = sim.start("gen9randombattle")
        print(f"Turn {b1.turn}")
        print(f"  P1: {b1.active_pokemon}  moves: {[m.id for m in b1.available_moves]}")
        print(f"  P2: {b2.active_pokemon}  moves: {[m.id for m in b2.available_moves]}")

        turns = 0
        while True:
            c1 = random_choice(b1) if sim.needs_choice_p1 else None
            c2 = random_choice(b2) if sim.needs_choice_p2 else None
            done, winner = sim.step(c1, c2)
            turns += 1
            if done:
                print(f"Done after {turns} turns. Winner: {winner}")
                break
            if turns % 10 == 0:
                hp = [(p.species, round(p.current_hp_fraction, 2)) for p in b1.team.values()]
                print(f"Turn {b1.turn}: {hp}")

    print("\nSequential benchmark (10 battles)...")
    t0 = time.perf_counter()
    total_steps = 0
    for i in range(10):
        result = run_random_battle()
        total_steps += result["steps"]
    elapsed = time.perf_counter() - t0
    print(f"  {10/elapsed:.1f} battles/sec")
    print(f"  {total_steps/elapsed:.0f} steps/sec")
    print(f"  {total_steps/10:.1f} avg steps/battle")
    print(f"  {elapsed/10*1000:.0f} ms/battle")
