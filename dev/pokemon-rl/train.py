"""
PPO self-play training loop for Pokemon RL.

Each env runs p1 (agent) vs p2 (random opponent).
Self-play against a frozen copy of the policy can be added later.

Run locally:
    uv run python train.py

Key hyperparameters (edit CONFIG below or pass via CLI):
    n_envs      — parallel envs (threads)
    horizon     — steps per rollout per env
    total_steps — total env steps to train for
    lr          — Adam learning rate
    n_epochs    — PPO epochs per rollout
    n_minibatch — minibatches per epoch
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from obs import obs_dim
from policy import Policy
from vecenv import ThreadedVecEnv

log = logging.getLogger("train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

ACTION_DIM = 26


@dataclass
class Config:
    format_id: str = "gen9randombattle"
    n_envs: int = 32
    horizon: int = 64         # steps collected per env per rollout
    total_steps: int = 10_000_000
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    n_minibatch: int = 4
    hidden_size: int = 256
    lstm_size: int = 256
    device: str = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10    # log every N rollouts


def compute_gae(
    rewards: np.ndarray,   # [T, N]
    values: np.ndarray,    # [T, N]
    dones: np.ndarray,     # [T, N]
    last_value: np.ndarray,  # [N]
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    T, N = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        next_done = dones[t]
        delta = rewards[t] + gamma * next_val * (1.0 - next_done) - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - next_done) * last_gae
        advantages[t] = last_gae
    return advantages


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set logits of invalid actions to -inf. Fall back to unmasked if all invalid."""
    any_valid = mask.any(dim=-1, keepdim=True)
    safe_mask = torch.where(any_valid, mask, torch.ones_like(mask))
    return logits.masked_fill(safe_mask == 0, float("-inf"))


def train(cfg: Config):
    device = torch.device(cfg.device)
    log.info(f"device={cfg.device} n_envs={cfg.n_envs} horizon={cfg.horizon}")

    vec = ThreadedVecEnv(cfg.n_envs, format_id=cfg.format_id)
    policy = Policy(
        obs_dim=obs_dim(),
        action_dim=ACTION_DIM,
        hidden_size=cfg.hidden_size,
        lstm_size=cfg.lstm_size,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)

    Path(cfg.checkpoint_dir).mkdir(exist_ok=True)

    # Rollout buffers — pre-allocate
    T, N = cfg.horizon, cfg.n_envs
    obs_buf    = np.zeros((T, N, obs_dim()), dtype=np.float32)
    act_buf    = np.zeros((T, N), dtype=np.int64)
    rew_buf    = np.zeros((T, N), dtype=np.float32)
    done_buf   = np.zeros((T, N), dtype=np.float32)
    val_buf    = np.zeros((T, N), dtype=np.float32)
    logp_buf   = np.zeros((T, N), dtype=np.float32)
    mask_buf   = np.zeros((T, N, ACTION_DIM), dtype=np.int8)

    obs_np, masks_np = vec.reset()
    h, c = policy.initial_state(N, device)
    # Keep LSTM state on CPU for buffer storage; move to device per rollout
    h0_buf = torch.zeros(T, N, cfg.lstm_size)
    c0_buf = torch.zeros(T, N, cfg.lstm_size)

    global_step = 0
    rollout = 0
    episode_rewards: list[float] = []
    wins = losses = 0

    t_start = time.perf_counter()

    while global_step < cfg.total_steps:
        rollout += 1
        t_rollout = time.perf_counter()

        # ----------------------------------------------------------------
        # Collect rollout
        # ----------------------------------------------------------------
        policy.eval()
        with torch.no_grad():
            for t in range(T):
                obs_t   = torch.from_numpy(obs_np).to(device)
                mask_t  = torch.from_numpy(masks_np).to(device)

                logits, value, (h, c) = policy(obs_t, (h, c))
                logits = masked_logits(logits, mask_t)
                dist   = Categorical(logits=logits)
                action = dist.sample()
                logp   = dist.log_prob(action)

                obs_buf[t]  = obs_np
                act_buf[t]  = action.cpu().numpy()
                val_buf[t]  = value.cpu().numpy()
                logp_buf[t] = logp.cpu().numpy()
                mask_buf[t] = masks_np
                h0_buf[t]   = h.cpu()
                c0_buf[t]   = c.cpu()

                obs_np, rew_np, done_np, masks_np = vec.step(action.cpu().numpy())
                rew_buf[t]  = rew_np
                done_buf[t] = done_np.astype(np.float32)

                # Track episode outcomes (only terminal steps have nonzero reward)
                for i in range(N):
                    if done_np[i]:
                        episode_rewards.append(rew_np[i])
                        if rew_np[i] > 0:
                            wins += 1
                        elif rew_np[i] < 0:
                            losses += 1

                # Reset LSTM state for finished envs
                done_t = torch.from_numpy(done_np).float().to(device)
                h = h * (1.0 - done_t).unsqueeze(1)
                c = c * (1.0 - done_t).unsqueeze(1)

                global_step += N

            # Bootstrap value for last step
            last_val = value.cpu().numpy()  # already computed at last t

        # ----------------------------------------------------------------
        # GAE
        # ----------------------------------------------------------------
        advantages = compute_gae(rew_buf, val_buf, done_buf, last_val, cfg.gamma, cfg.gae_lambda)
        returns = advantages + val_buf

        # Flatten [T, N] -> [T*N]
        obs_flat   = obs_buf.reshape(T * N, -1)
        act_flat   = act_buf.reshape(T * N)
        adv_flat   = advantages.reshape(T * N)
        ret_flat   = returns.reshape(T * N)
        logp_flat  = logp_buf.reshape(T * N)
        mask_flat  = mask_buf.reshape(T * N, ACTION_DIM)
        done_flat  = done_buf  # keep [T, N] for LSTM unroll
        h0_flat    = h0_buf[0]  # [N, lstm_size] — state at start of rollout
        c0_flat    = c0_buf[0]

        # Normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # ----------------------------------------------------------------
        # PPO update
        # ----------------------------------------------------------------
        policy.train()
        batch_size = T * N
        minibatch_size = batch_size // cfg.n_minibatch

        pg_losses = []
        vf_losses = []
        ent_losses = []

        for _ in range(cfg.n_epochs):
            idx = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb = idx[start:start + minibatch_size]

                # For LSTM we need full sequences — use full-batch sequence unroll
                # (minibatch over sequences would require storing per-step states;
                # for simplicity we recompute over the full rollout each epoch)
                break  # handled below

            # Full-sequence recompute (simpler and correct for LSTM)
            obs_seq  = torch.from_numpy(obs_flat.reshape(T, N, -1)).to(device)
            done_seq = torch.from_numpy(done_flat).to(device)
            logits_new, values_new = policy.forward_sequence(
                obs_seq, done_seq,
                h0_flat.to(device), c0_flat.to(device),
            )

            mask_t2  = torch.from_numpy(mask_flat).to(device)
            logits_new = masked_logits(logits_new, mask_t2)
            dist_new   = Categorical(logits=logits_new)

            act_t    = torch.from_numpy(act_flat).long().to(device)
            logp_new = dist_new.log_prob(act_t)
            entropy  = dist_new.entropy().mean()

            logp_old = torch.from_numpy(logp_flat).to(device)
            adv_t    = torch.from_numpy(adv_flat).to(device)
            ret_t    = torch.from_numpy(ret_flat).to(device)

            ratio     = torch.exp(logp_new - logp_old)
            pg_loss1  = -adv_t * ratio
            pg_loss2  = -adv_t * ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            pg_loss   = torch.max(pg_loss1, pg_loss2).mean()
            vf_loss   = F.mse_loss(values_new, ret_t)
            loss      = pg_loss + cfg.value_coef * vf_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            pg_losses.append(pg_loss.item())
            vf_losses.append(vf_loss.item())
            ent_losses.append(entropy.item())

        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------
        if rollout % cfg.log_interval == 0:
            elapsed = time.perf_counter() - t_start
            sps = global_step / elapsed
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else float("nan")
            mean_ep_rew = np.mean(episode_rewards[-100:]) if episode_rewards else float("nan")
            log.info(
                f"step={global_step:,} rollout={rollout} sps={sps:.0f} "
                f"win_rate={win_rate:.3f} mean_ep_rew={mean_ep_rew:.3f} "
                f"pg={np.mean(pg_losses):.4f} vf={np.mean(vf_losses):.4f} "
                f"ent={np.mean(ent_losses):.4f}"
            )

        # Save checkpoint every 100 rollouts
        if rollout % 100 == 0:
            ckpt = Path(cfg.checkpoint_dir) / f"policy_{global_step:010d}.pt"
            torch.save({"policy": policy.state_dict(), "step": global_step}, ckpt)
            log.info(f"saved {ckpt}")

    vec.close()
    log.info(f"training done: {global_step:,} steps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=10_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--format", type=str, default="gen9randombattle")
    args = parser.parse_args()

    cfg = Config(
        n_envs=args.n_envs,
        horizon=args.horizon,
        total_steps=args.total_steps,
        lr=args.lr,
        format_id=args.format,
    )
    if args.device:
        cfg.device = args.device

    train(cfg)


if __name__ == "__main__":
    main()
