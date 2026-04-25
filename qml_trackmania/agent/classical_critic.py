"""
agent/classical_critic.py
=========================
Classical Twin-Q Critics for SAC.

We intentionally keep critics classical for two reasons:
  1. Training stability — quantum gradients are noisier
  2. Demo clarity — quantum appears only in the actor (policy)

The twin-critic design (two independent Q-networks, take min) is
the standard SAC trick to prevent Q-value overestimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from configs.qml_config import EnvConfig, CriticConfig


class QNetwork(nn.Module):
    """Single Q(s, a) → scalar network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        sizes = [obs_dim + act_dim] + list(hidden_sizes) + [1]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs    : (batch, obs_dim)
        action : (batch, act_dim)

        Returns
        -------
        q : (batch, 1)
        """
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class TwinCritic(nn.Module):
    """
    Twin-Q critic: two independent Q-networks.
    During training, target = min(Q1, Q2) to combat overestimation.
    """

    def __init__(self, env_cfg: EnvConfig, critic_cfg: CriticConfig):
        super().__init__()
        self.q1 = QNetwork(env_cfg.obs_dim, env_cfg.act_dim, critic_cfg.hidden_sizes)
        self.q2 = QNetwork(env_cfg.obs_dim, env_cfg.act_dim, critic_cfg.hidden_sizes)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (Q1, Q2) values — both needed for critic loss."""
        return self.q1(obs, action), self.q2(obs, action)

    def min_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return min(Q1, Q2) — used for actor loss and target computation."""
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)
