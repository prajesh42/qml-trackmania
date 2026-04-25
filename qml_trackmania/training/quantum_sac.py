"""
training/quantum_sac.py
=======================
Main Quantum-SAC (QML-SAC) training loop.

This integrates the hybrid quantum actor + classical critics with the
TMRL Gymnasium environment interface.

Usage
-----
  # Start TMRL server & worker first (as normal), then:
  python training/quantum_sac.py

Algorithm
---------
  1. Collect transition from TMRL env using quantum actor
  2. Store in replay buffer
  3. After warmup, every `update_every` steps:
       a. Sample mini-batch
       b. Compute TD target:  y = r + γ(1-d)[min_Q(s',a') - α·log π(a'|s')]
       c. Update critics:     min ||Q(s,a) - y||²
       d. Update actor:       max E[min_Q(s,a') - α·log π(a'|s)]
       e. Auto-tune entropy:  update α so H(π) ≈ target_entropy
       f. Soft-update target critics
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from typing import Optional
import logging

# Add both project root and qml_trackmania/ to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ibm-trackmania/
_QML  = os.path.join(_ROOT, "qml_trackmania")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _QML)

from configs.qml_config import CFG, Config
from agent.quantum_actor import QuantumActor
from agent.classical_critic import TwinCritic
from agent.replay_buffer import ReplayBuffer
from training.tmrl_integration import make_tmrl_env
from utils.logger import TrainingLogger

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)


class QuantumSAC:
    """
    Quantum Soft Actor-Critic agent.

    Parameters
    ----------
    cfg : Config
        Full configuration object (env + quantum + sac + critic + train).
    device : str
        "cpu" or "cuda". Note: quantum circuits always run on CPU (Aer),
        but classical network tensors can live on GPU.
    """

    def __init__(self, cfg: Config, device: str = "cpu"):
        self.cfg = cfg
        self.device = device

        log.info("Initialising Quantum Actor...")
        self.actor = QuantumActor(cfg.env, cfg.quantum).to(device)

        log.info("Initialising Twin Critics...")
        self.critic        = TwinCritic(cfg.env, cfg.critic).to(device)
        self.critic_target = deepcopy(self.critic).to(device)

        # Freeze target — updated via soft copy only
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # Replay buffer
        self.buffer = ReplayBuffer(
            cfg.env.obs_dim,
            cfg.env.act_dim,
            cfg.sac.buffer_size,
            device,
        )

        # Optimisers
        # Note: actor optimizer covers both classical + quantum params
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=cfg.sac.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.sac.critic_lr)

        # Automatic entropy tuning
        if cfg.sac.auto_alpha:
            self.target_entropy = cfg.sac.target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.sac.alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.log_alpha = None
            self.alpha = cfg.sac.alpha

        self.total_steps = 0

        # Print parameter summary (useful for IBM demo)
        param_counts = self.actor.num_parameters()
        log.info("Actor parameter breakdown:")
        for k, v in param_counts.items():
            log.info(f"  {k}: {v}")

    # ─────────────────────────────────────────────────────────────────────────
    # Action selection
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Convert obs → action for environment interaction."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _, _ = self.actor.get_action(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    # ─────────────────────────────────────────────────────────────────────────
    # SAC update step
    # ─────────────────────────────────────────────────────────────────────────

    def update(self) -> dict:
        """
        One gradient update step (critics + actor + alpha).
        Returns a dict of scalar losses for logging.
        """
        cfg = self.cfg.sac
        obs, actions, rewards, next_obs, dones = self.buffer.sample(cfg.batch_size)

        # ── 1. Critic update ──────────────────────────────────────────────────
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.get_action(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            min_q_next = torch.min(q1_next, q2_next)
            # TD target (soft Bellman)
            target_q = rewards + cfg.gamma * (1.0 - dones) * (
                min_q_next - self.alpha * next_log_pi
            )

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        self.critic_opt.step()

        # ── 2. Actor update ───────────────────────────────────────────────────
        # Freeze critic params during actor update (efficiency)
        for p in self.critic.parameters():
            p.requires_grad_(False)

        new_actions, log_pi, _ = self.actor.get_action(obs)
        min_q = self.critic.min_q(obs, new_actions)
        actor_loss = (self.alpha * log_pi - min_q).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        # Clip quantum gradients (can be large with param-shift)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        for p in self.critic.parameters():
            p.requires_grad_(True)

        # ── 3. Entropy temperature (alpha) update ─────────────────────────────
        alpha_loss = torch.tensor(0.0)
        if self.log_alpha is not None:
            with torch.no_grad():
                _, log_pi_new, _ = self.actor.get_action(obs)
            alpha_loss = -(
                self.log_alpha.exp() * (log_pi_new + self.target_entropy)
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        # ── 4. Soft target update ─────────────────────────────────────────────
        tau = cfg.tau
        with torch.no_grad():
            for p, p_tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_tgt.data.mul_(1.0 - tau)
                p_tgt.data.add_(tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha_loss":  alpha_loss.item(),
            "alpha":       self.alpha,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint management
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor_state_dict":         self.actor.state_dict(),
            "critic_state_dict":        self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_opt_state_dict":     self.actor_opt.state_dict(),
            "critic_opt_state_dict":    self.critic_opt.state_dict(),
            "log_alpha":                self.log_alpha,
            "total_steps":              self.total_steps,
        }, path)
        log.info(f"Checkpoint saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.critic_target.load_state_dict(ckpt["critic_target_state_dict"])
        self.actor_opt.load_state_dict(ckpt["actor_opt_state_dict"])
        self.critic_opt.load_state_dict(ckpt["critic_opt_state_dict"])
        if self.log_alpha is not None and ckpt["log_alpha"] is not None:
            self.log_alpha.data = ckpt["log_alpha"].data
            self.alpha = self.log_alpha.exp().item()
        self.total_steps = ckpt["total_steps"]
        log.info(f"Checkpoint loaded ← {path} (step {self.total_steps})")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Config = CFG, resume_path: Optional[str] = None):
    """
    Main training loop.

    Integrates with TMRL's Gymnasium interface for sample collection,
    and runs QML-SAC gradient updates.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # Environment
    env = make_tmrl_env(cfg)
    log.info(f"Environment: obs_dim={cfg.env.obs_dim}, act_dim={cfg.env.act_dim}")

    # Agent
    agent = QuantumSAC(cfg, device)
    if resume_path:
        agent.load(resume_path)

    # Logger
    logger = TrainingLogger(cfg.train)

    # ── Training loop ──────────────────────────────────────────────────────
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_steps  = 0
    episode_num    = 0
    t_start        = time.time()

    log.info(f"Starting training — warmup for {cfg.sac.warmup_steps} steps...")

    for step in range(cfg.train.max_steps):
        agent.total_steps = step

        # Action selection
        if step < cfg.sac.warmup_steps:
            action = env.action_space.sample()           # random warmup
        else:
            action = agent.select_action(obs)            # quantum policy

        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        agent.buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        episode_reward += reward
        episode_steps  += 1

        # Gradient updates (after warmup)
        losses = {}
        if step >= cfg.sac.warmup_steps and step % cfg.sac.update_every == 0:
            for _ in range(cfg.sac.updates_per_step):
                losses = agent.update()

        # Episode end
        if done:
            episode_num += 1
            log.info(
                f"Episode {episode_num:4d} | "
                f"Steps {step:7d} | "
                f"Reward {episode_reward:8.2f} | "
                f"Ep-len {episode_steps:4d} | "
                f"α {agent.alpha:.4f}"
            )
            logger.log_episode(step, episode_reward, episode_steps, losses)

            obs, _ = env.reset()
            episode_reward = 0.0
            episode_steps  = 0

        # Checkpoint
        if step > 0 and step % cfg.train.save_every == 0:
            ckpt_path = os.path.join(
                cfg.train.checkpoint_dir, f"{cfg.train.run_name}_step{step}.pt"
            )
            agent.save(ckpt_path)

    log.info("Training complete!")
    env.close()
    logger.close()


if __name__ == "__main__":
    train()