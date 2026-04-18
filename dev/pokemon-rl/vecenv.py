"""
Threaded vectorized environment.

Runs N PokemonEnv instances concurrently in a ThreadPoolExecutor.
Each env's step() blocks on Node I/O (readline), so threads spend most
of their time waiting — the GIL releases during blocking I/O, allowing
true concurrency for the Node-bound portion of each step.

The Python parse_message work (~0.9ms/step) does hold the GIL briefly,
but at 10-100x less than the Node I/O wait it's not the bottleneck.

Interface mirrors standard vecenv:
    obs, masks = vec.reset()
    obs, rewards, dones, masks = vec.step(actions)
    vec.close()

All arrays are numpy float32/int8, shape (N, ...).
"""

from __future__ import annotations

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from env import PokemonEnv
from obs import obs_dim

ACTION_SPACE_SIZE = 26  # Gen 9 singles


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
        self._obs_buf   = np.zeros((n_envs, obs_dim()), dtype=np.float32)
        self._mask_buf  = np.ones((n_envs, ACTION_SPACE_SIZE), dtype=np.int8)
        self._rew_buf   = np.zeros(n_envs, dtype=np.float32)
        self._done_buf  = np.zeros(n_envs, dtype=bool)

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Reset all envs concurrently. Returns (obs [N, obs_dim], masks [N, 26])."""
        def _reset(i):
            obs, info = self._envs[i].reset()
            return i, obs, info["action_mask"]

        futures = [self._executor.submit(_reset, i) for i in range(self.n_envs)]
        for f in as_completed(futures):
            i, obs, mask = f.result()
            self._obs_buf[i]  = obs
            self._mask_buf[i] = mask

        return self._obs_buf.copy(), self._mask_buf.copy()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Step all envs with their respective actions.
        Envs that finish are automatically reset.
        Returns (obs, rewards, dones, masks) each shape (N, ...).
        """
        def _step(i, action):
            obs, reward, done, _, info = self._envs[i].step(int(action))
            if done:
                obs, reset_info = self._envs[i].reset()
                mask = reset_info["action_mask"]
            else:
                mask = info["action_mask"]
            return i, obs, reward, done, mask

        futures = [
            self._executor.submit(_step, i, actions[i])
            for i in range(self.n_envs)
        ]
        for f in as_completed(futures):
            i, obs, reward, done, mask = f.result()
            self._obs_buf[i]  = obs
            self._rew_buf[i]  = reward
            self._done_buf[i] = done
            self._mask_buf[i] = mask

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
