"""
PPO policy: MLP encoder -> LSTM -> linear heads (logits + value).

Usage:
    policy = Policy(obs_dim=132, action_dim=26, hidden_size=256, lstm_size=256)
    # Single step (eval):
    logits, value, (h, c) = policy(obs_tensor, (h, c))
    # Sequence (train):
    logits, values = policy.forward_sequence(obs_seq, dones_seq, initial_state)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Policy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        lstm_size: int = 256,
    ):
        super().__init__()
        self.lstm_size = lstm_size

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTMCell(hidden_size, lstm_size)
        self.actor  = nn.Linear(lstm_size, action_dim)
        self.critic = nn.Linear(lstm_size, 1)

        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        # Scale actor output small to start near-uniform policy
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def initial_state(self, batch_size: int, device: torch.device):
        z = torch.zeros(batch_size, self.lstm_size, device=device)
        return (z, z)

    def forward(
        self,
        obs: Tensor,           # [B, obs_dim]
        state: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """Single-step forward for rollout collection."""
        h, c = state
        enc = self.encoder(obs)
        h, c = self.lstm(enc, (h, c))
        logits = self.actor(h)
        value  = self.critic(h).squeeze(-1)
        return logits, value, (h, c)

    def forward_sequence(
        self,
        obs: Tensor,           # [T, B, obs_dim]
        dones: Tensor,         # [T, B] float — 1.0 on terminal steps
        h0: Tensor,            # [B, lstm_size]
        c0: Tensor,            # [B, lstm_size]
    ) -> tuple[Tensor, Tensor]:
        """
        Unroll LSTM over a horizon T, resetting hidden state on episode boundaries.
        Returns logits [T*B, action_dim] and values [T*B].
        """
        T, B = obs.shape[:2]
        enc = self.encoder(obs.reshape(T * B, -1)).reshape(T, B, -1)

        logits_list = []
        values_list = []
        h, c = h0, c0

        for t in range(T):
            # Reset state for envs that finished last step
            mask = 1.0 - dones[t]          # [B]
            h = h * mask.unsqueeze(1)
            c = c * mask.unsqueeze(1)
            h, c = self.lstm(enc[t], (h, c))
            logits_list.append(self.actor(h))
            values_list.append(self.critic(h).squeeze(-1))

        logits = torch.stack(logits_list, dim=0).reshape(T * B, -1)
        values = torch.stack(values_list, dim=0).reshape(T * B)
        return logits, values
