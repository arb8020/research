"""
Process-based vectorized environment.

Each env runs in a dedicated worker process — no GIL sharing. The GIL
is the bottleneck because Battle.parse_message is pure Python CPU work
(~0.9ms/step) with almost no I/O release window.

Each worker process:
  - Imports torch, poke-env, obs encoder once at startup
  - Owns one PokemonEnv and one Node subprocess (persistent across battles)
  - Communicates via multiprocessing.Queue (action in, result out)

Startup is slow (~5-10s for all workers to initialize) but steady-state
throughput is N× single-process rate since there's no GIL contention.

Interface:
    obs, masks = vec.reset()
    obs, rewards, dones, masks = vec.step(actions)
    vec.close()

All returned arrays are numpy float32/int8, shape (N, ...).
"""

from __future__ import annotations

import multiprocessing as mp
import numpy as np
from typing import Optional

ACTION_SPACE_SIZE = 26


def _worker(
    worker_id: int,
    format_id: str,
    action_q: mp.Queue,
    result_q: mp.Queue,
    env_vars: dict,
):
    """Worker process: owns one env, loops on action_q, writes to result_q."""
    import os
    os.environ.update(env_vars)

    # Imports happen once per worker — heavy but amortized
    from env import PokemonEnv
    from obs import obs_dim

    env = PokemonEnv(format_id=format_id)
    obs, info = env.reset()
    # Send initial obs after reset
    result_q.put((worker_id, obs, info["action_mask"], None, False))

    while True:
        action = action_q.get()
        if action is None:  # shutdown sentinel
            env.close()
            return
        obs, reward, done, _, info = env.step(int(action))
        if done:
            obs, reset_info = env.reset()
            mask = reset_info["action_mask"]
        else:
            mask = info["action_mask"]
        result_q.put((worker_id, obs, mask, reward, done))


class ProcessVecEnv:
    def __init__(
        self,
        n_envs: int,
        format_id: str = "gen9randombattle",
        env_vars: dict | None = None,
    ):
        self.n_envs = n_envs
        from obs import obs_dim as _obs_dim
        self._obs_dim = _obs_dim()

        self._obs_buf  = np.zeros((n_envs, self._obs_dim), dtype=np.float32)
        self._mask_buf = np.ones((n_envs, ACTION_SPACE_SIZE), dtype=np.int8)
        self._rew_buf  = np.zeros(n_envs, dtype=np.float32)
        self._done_buf = np.zeros(n_envs, dtype=bool)

        ctx = mp.get_context("spawn")
        self._result_q: mp.Queue = ctx.Queue()
        self._action_qs: list[mp.Queue] = [ctx.Queue() for _ in range(n_envs)]

        import os
        _env_vars = {k: os.environ[k] for k in ("NODE_BIN", "SHOWDOWN_PATH") if k in os.environ}
        if env_vars:
            _env_vars.update(env_vars)

        self._procs = []
        for i in range(n_envs):
            p = ctx.Process(
                target=_worker,
                args=(i, format_id, self._action_qs[i], self._result_q, _env_vars),
                daemon=True,
            )
            p.start()
            self._procs.append(p)

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Collect initial observations from all workers (sent on startup)."""
        for _ in range(self.n_envs):
            worker_id, obs, mask, _, _ = self._result_q.get()
            self._obs_buf[worker_id]  = obs
            self._mask_buf[worker_id] = mask
        return self._obs_buf.copy(), self._mask_buf.copy()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Dispatch actions to all workers
        for i in range(self.n_envs):
            self._action_qs[i].put(int(actions[i]))
        # Collect results
        for _ in range(self.n_envs):
            worker_id, obs, mask, reward, done = self._result_q.get()
            self._obs_buf[worker_id]  = obs
            self._mask_buf[worker_id] = mask
            self._rew_buf[worker_id]  = reward
            self._done_buf[worker_id] = done
        return (
            self._obs_buf.copy(),
            self._rew_buf.copy(),
            self._done_buf.copy(),
            self._mask_buf.copy(),
        )

    def close(self):
        for q in self._action_qs:
            q.put(None)  # shutdown sentinel
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
