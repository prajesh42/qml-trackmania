"""
Quantum-ready policy/value models for TrackMania LIDAR pipelines.

These models are "hybrid" and can run fully in torch out of the box.
The feature mapping uses phase/amplitude style trigonometric blocks so the
module is compatible with real-time constraints while remaining a good place
to plug a true quantum backend later.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from tmrl.actor import TorchActorModule
from tmrl.util import prod


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def _obs_dim_and_mode(observation_space):
    try:
        dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
        tuple_obs = True
    except TypeError:
        dim_obs = prod(observation_space.shape)
        tuple_obs = False
    return dim_obs, tuple_obs


class HybridQuantumFeatureMap(nn.Module):
    """
    Lightweight trigonometric feature map.

    This is intentionally deterministic and torch-native for low-latency use in
    real-time control loops. It behaves like a quantum-inspired embedding layer.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, depth: int = 2):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be > 0, got {depth}")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.phase_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(depth))
        self.amp_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(depth))
        self.norm_layers = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(depth))

    def forward(self, x):
        x = self.input_proj(x)
        for phase, amp, norm in zip(self.phase_layers, self.amp_layers, self.norm_layers):
            update = torch.sin(phase(x)) * torch.cos(amp(x))
            x = norm(x + update)
        return x


class QuantumSquashedGaussianMLPActor(TorchActorModule):
    """
    Drop-in replacement for SquashedGaussianMLPActor with a quantum-ready backbone.
    """
    def __init__(self, observation_space, action_space, hidden_dim: int = 256, depth: int = 2):
        super().__init__(observation_space, action_space)
        dim_obs, tuple_obs = _obs_dim_and_mode(observation_space)
        self.tuple_obs = tuple_obs

        dim_act = action_space.shape[0]
        self.act_limit = action_space.high[0]

        self.net = HybridQuantumFeatureMap(input_dim=dim_obs, hidden_dim=hidden_dim, depth=depth)
        self.mu_layer = nn.Linear(hidden_dim, dim_act)
        self.log_std_layer = nn.Linear(hidden_dim, dim_act)

    def _flatten_obs(self, obs):
        return torch.cat(obs, -1) if self.tuple_obs else torch.flatten(obs, start_dim=1)

    def forward(self, obs, test=False, with_logprob=True):
        x = self._flatten_obs(obs)
        x = self.net(x)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        pi_action = mu if test else pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            action, _ = self.forward(obs, test=test, with_logprob=False)
            res = action.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


class QuantumMLPQFunction(nn.Module):
    def __init__(self, obs_space, act_space, hidden_dim: int = 256, depth: int = 2):
        super().__init__()
        obs_dim, tuple_obs = _obs_dim_and_mode(obs_space)
        self.tuple_obs = tuple_obs
        act_dim = act_space.shape[0]
        self.net = HybridQuantumFeatureMap(input_dim=obs_dim + act_dim, hidden_dim=hidden_dim, depth=depth)
        self.q_out = nn.Linear(hidden_dim, 1)

    def forward(self, obs, act):
        if self.tuple_obs:
            x = torch.cat((*obs, act), -1)
        else:
            x = torch.cat((torch.flatten(obs, start_dim=1), act), -1)
        q = self.q_out(self.net(x))
        return torch.squeeze(q, -1)


class QuantumMLPActorCritic(nn.Module):
    """
    Actor-critic pair compatible with SpinupSacAgent.
    """
    def __init__(self, observation_space, action_space, hidden_dim: int = 256, depth: int = 2):
        super().__init__()
        self.actor = QuantumSquashedGaussianMLPActor(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dim=hidden_dim,
            depth=depth,
        )
        self.q1 = QuantumMLPQFunction(
            obs_space=observation_space,
            act_space=action_space,
            hidden_dim=hidden_dim,
            depth=depth,
        )
        self.q2 = QuantumMLPQFunction(
            obs_space=observation_space,
            act_space=action_space,
            hidden_dim=hidden_dim,
            depth=depth,
        )

    def act(self, obs, test=False):
        with torch.no_grad():
            action, _ = self.actor(obs, test=test, with_logprob=False)
            res = action.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res
