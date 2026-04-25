"""
agent/quantum_actor.py
======================
Hybrid Quantum-Classical Actor for Soft Actor-Critic.

Architecture:
    obs (22-dim)
      │
      ▼
    Pre-encoder: Linear(22 → 4) + Tanh   ← classical
      │  rescale to [-π, π]
      ▼
    Angle Encoding: RY gates              ← quantum (encoding)
      │
      ▼
    RealAmplitudes Ansatz (4q, 2 reps)   ← quantum (trainable PQC)
      │  EstimatorQNN → ⟨Z_i⟩ ∈ [-1,1]
      ▼
    Post-head: Linear(4 → 2*3) + split  ← classical
      │
      ├── mean    (3-dim) → tanh → action mean
      └── log_std (3-dim) → clamp → action log std

The actor outputs a squashed Gaussian policy (as in standard SAC).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional

from quantum.qnn import QuantumLayer
from configs.qml_config import QuantumConfig, EnvConfig

LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0
EPSILON = 1e-6


class QuantumActor(nn.Module):
    """
    Hybrid Quantum-Classical Stochastic Policy.

    The policy π_θ(a|s) is a squashed Gaussian:
        mean, log_std = f_quantum(s)
        a = tanh(mean + std * ε),  ε ~ N(0, I)

    Parameters
    ----------
    env_cfg : EnvConfig
    q_cfg   : QuantumConfig
    """

    def __init__(self, env_cfg: EnvConfig, q_cfg: QuantumConfig):
        super().__init__()
        self.obs_dim = env_cfg.obs_dim
        self.act_dim = env_cfg.act_dim
        self.n_qubits = q_cfg.n_qubits

        # ── Classical Pre-encoder ────────────────────────────────────────────
        # Compresses obs_dim → n_qubits for angle encoding.
        # Output rescaled to [-π, π] via Tanh + π scaling.
        if q_cfg.use_pre_encoder:
            self.pre_encoder = nn.Sequential(
                nn.Linear(env_cfg.obs_dim, q_cfg.pre_encoder_hidden),
                nn.ReLU(),
                nn.Linear(q_cfg.pre_encoder_hidden, q_cfg.n_qubits),
                nn.Tanh(),   # output ∈ (-1, 1), will be scaled to (-π, π)
            )
        else:
            assert env_cfg.obs_dim == q_cfg.n_qubits, (
                "Without pre-encoder, obs_dim must equal n_qubits."
            )
            self.pre_encoder = nn.Identity()

        # ── Quantum Layer ────────────────────────────────────────────────────
        self.quantum_layer = QuantumLayer(q_cfg)

        # ── Classical Post-head ──────────────────────────────────────────────
        # Maps n_qubits expectation values → 2 * act_dim (mean + log_std)
        if q_cfg.use_post_head:
            self.post_head = nn.Linear(q_cfg.n_qubits, 2 * env_cfg.act_dim)
        else:
            assert q_cfg.n_qubits == 2 * env_cfg.act_dim, (
                "Without post-head, n_qubits must equal 2 * act_dim."
            )
            self.post_head = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Xavier init for classical layers."""
        for m in [self.pre_encoder, self.post_head]:
            for layer in (m.modules() if hasattr(m, 'modules') else [m]):
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass — returns action mean and log_std.

        Parameters
        ----------
        obs : Tensor (batch, obs_dim)

        Returns
        -------
        mean    : Tensor (batch, act_dim)  — pre-squash action mean
        log_std : Tensor (batch, act_dim)  — log standard deviation
        """
        # 1. Classical pre-encoding
        encoded = self.pre_encoder(obs)            # (batch, n_qubits) ∈ (-1, 1)
        encoded = encoded * math.pi                # scale to (-π, π) for RY angles

        # 2. Quantum processing (PQC → Pauli-Z expectation values)
        q_out = self.quantum_layer(encoded)        # (batch, n_qubits) ∈ (-1, 1)

        # 3. Classical post-head
        out = self.post_head(q_out)                # (batch, 2 * act_dim)
        mean, log_std = out.chunk(2, dim=-1)

        # Clamp log_std for numerical stability
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action and compute log-probability (for SAC update).

        Returns
        -------
        action      : Tensor (batch, act_dim) ∈ [-1, 1]  (after tanh squash)
        log_prob    : Tensor (batch, 1)
        mean_action : Tensor (batch, act_dim)             (for deterministic eval)
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        if deterministic:
            # Use mean action (for evaluation / benchmark)
            action = torch.tanh(mean)
            log_prob = torch.zeros(mean.shape[0], 1, device=obs.device)
            return action, log_prob, action

        # Reparameterisation trick: a = tanh(μ + σ·ε), ε ~ N(0,I)
        normal = Normal(mean, std)
        x_t = normal.rsample()             # pre-squash sample
        action = torch.tanh(x_t)          # squashed to (-1, 1)

        # Log-prob with change-of-variables correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1.0 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # (batch, 1)

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action

    def num_parameters(self) -> dict:
        """Return parameter counts per component (useful for IBM slides)."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "pre_encoder (classical)": count(self.pre_encoder),
            "quantum_layer (PQC)":     self.quantum_layer.num_params,
            "post_head (classical)":   count(self.post_head),
            "total":                   count(self),
        }
