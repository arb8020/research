"""
Threaded vectorized environment.

Runs N PokemonEnv instances concurrently in a ThreadPoolExecutor.

Profiling shows the per-step breakdown is:
  - Node game logic (readline wait): ~0.76ms median  <- actual bottleneck
  - poke-env parse_block: ~0.10ms                    <- negligible
  - pipe write: ~0.002ms                             <- negligible

The GIL is not the bottleneck. parse_block holds the GIL for ~0.10ms
per step vs ~0.76ms blocked in readline (GIL released). Threading gives
close to N× throughput up to the CPU count of the machine.

Interface:
    obs, masks = vec.reset()
    obs, rewards, dones, masks = vec.step(actions)
    vec.close()

All arrays are numpy float32/int8, shape (N, ...).
"""

from __future__ import annotations

import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from env import PokemonEnv
from obs import obs_dim

ACTION_SPACE_SIZE = 26
STEP_TIMEOUT = 35.0   # seconds — slightly above the 30s readline timeout in sim_bridge
RESET_TIMEOUT = 120.0  # Node startup can be slow on cold container

log = logging.getLogger("vecenv")


class ThreadedVecEnv:
    def __init__(
        self,
        n_envs: int,
        format_id: str = "gen9randombattle",
        opponent_fn=None,
    ):
        self.n_envs = n_envs
        self._envs = [
            PokemonEnv(format_id=format_id, opponent_fn=opponent_fn)
            for _ in range(n_envs)
        ]
        self._executor = ThreadPoolExecutor(max_workers=n_envs)
        self._obs_buf  = np.zeros((n_envs, obs_dim()), dtype=np.float32)
        self._mask_buf = np.ones((n_envs, ACTION_SPACE_SIZE), dtype=np.int8)
        self._rew_buf  = np.zeros(n_envs, dtype=np.float32)
        self._done_buf = np.zeros(n_envs, dtype=bool)

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        def _reset(i):
            t0 = time.perf_counter()
            obs, info = self._envs[i].reset()
            log.debug("env %d reset in %.2fs", i, time.perf_counter() - t0)
            return i, obs, info["action_mask"]

        t_reset = time.perf_counter()
        log.info("resetting %d envs", self.n_envs)
        futures = {self._executor.submit(_reset, i): i for i in range(self.n_envs)}
        pending = set(futures.values())
        try:
            for f in as_completed(futures, timeout=RESET_TIMEOUT):
                i, obs, mask = f.result()
                pending.discard(i)
                self._obs_buf[i]  = obs
                self._mask_buf[i] = mask
                log.debug("env %d ready (%d pending)", i, len(pending))
        except TimeoutError:
            log.error("reset timed out after %.1fs — stuck envs: %s", RESET_TIMEOUT, sorted(pending))
            # Re-submit stuck envs with fresh state; they will retry on next reset
            for i in pending:
                self._envs[i].close()
                future = self._executor.submit(_reset, i)
                try:
                    idx, obs, mask = future.result(timeout=RESET_TIMEOUT)
                    self._obs_buf[idx]  = obs
                    self._mask_buf[idx] = mask
                except Exception:
                    log.error("env %d failed retry reset — using zeros", i)
        log.info("all envs reset in %.2fs", time.perf_counter() - t_reset)
        return self._obs_buf.copy(), self._mask_buf.copy()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        def _step(i, action):
            t0 = time.perf_counter()
            obs, reward, done, _, info = self._envs[i].step(int(action))
            if done:
                obs, reset_info = self._envs[i].reset()
                mask = reset_info["action_mask"]
            else:
                mask = info["action_mask"]
            elapsed = time.perf_counter() - t0
            if elapsed > 5.0:
                log.warning("env %d step took %.2fs (done=%s)", i, elapsed, done)
            return i, obs, reward, done, mask

        futures = {
            self._executor.submit(_step, i, actions[i]): i
            for i in range(self.n_envs)
        }
        pending = set(futures.values())
        try:
            for f in as_completed(futures, timeout=STEP_TIMEOUT):
                i, obs, reward, done, mask = f.result()
                pending.discard(i)
                self._obs_buf[i]  = obs
                self._rew_buf[i]  = reward
                self._done_buf[i] = done
                self._mask_buf[i] = mask
        except TimeoutError:
            log.error("step timed out — stuck envs: %s, resetting them", sorted(pending))
            for i in pending:
                self._envs[i].close()
                try:
                    obs, info = self._envs[i].reset()
                    self._obs_buf[i]  = obs
                    self._mask_buf[i] = info["action_mask"]
                except Exception:
                    log.error("env %d failed reset after step timeout", i)
        return (
            self._obs_buf.copy(),
            self._rew_buf.copy(),
            self._done_buf.copy(),
            self._mask_buf.copy(),
        )

    def close(self):
        self._executor.shutdown(wait=False)
        for env in self._envs:
            env.close()
