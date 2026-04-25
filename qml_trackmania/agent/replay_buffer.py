"""
agent/replay_buffer.py
======================
Off-policy experience replay buffer for SAC.
Compatible with TMRL's distributed data collection setup.
"""

import numpy as np
import torch
from typing import Tuple


class ReplayBuffer:
    """
    Circular experience replay buffer.

    Stores transitions (s, a, r, s', done) and samples random mini-batches.

    Parameters
    ----------
    obs_dim    : int
    act_dim    : int
    capacity   : int   — maximum number of transitions (default 500k)
    device     : str   — "cpu" or "cuda"
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int = 500_000,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # Pre-allocate storage as numpy arrays (faster than list of tensors)
        self.obs     = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim),  dtype=np.float32)
        self.rewards = np.zeros((capacity, 1),        dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones   = np.zeros((capacity, 1),        dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition to the buffer."""
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add a batch of transitions (for TMRL distributed workers)."""
        batch_size = obs.shape[0]
        indices = np.arange(self.ptr, self.ptr + batch_size) % self.capacity
        self.obs[indices]      = obs
        self.actions[indices]  = actions
        self.rewards[indices]  = rewards.reshape(-1, 1)
        self.next_obs[indices] = next_obs
        self.dones[indices]    = dones.reshape(-1, 1)
        self.ptr  = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int = 256) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random mini-batch.

        Returns
        -------
        (obs, actions, rewards, next_obs, dones) — all Tensors on self.device
        """
        assert self.size >= batch_size, (
            f"Buffer has {self.size} samples, need {batch_size} to sample."
        )
        idx = np.random.randint(0, self.size, size=batch_size)

        def to_tensor(x):
            return torch.tensor(x[idx], dtype=torch.float32).to(self.device)

        return (
            to_tensor(self.obs),
            to_tensor(self.actions),
            to_tensor(self.rewards),
            to_tensor(self.next_obs),
            to_tensor(self.dones),
        )

    def __len__(self) -> int:
        return self.size
