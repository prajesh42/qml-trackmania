"""
utils/benchmark.py
==================
Compare Classical SAC actor vs Quantum SAC actor.

Metrics collected:
  - Inference latency (ms per action)
  - Parameter count
  - Episode reward over N rollouts (MockTMEnv)
  - Circuit depth and gate count

Run:
    python utils/benchmark.py
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_QML  = os.path.join(_ROOT, "qml_trackmania")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _QML)

import time
import numpy as np
import torch
import torch.nn as nn
from typing import List

from configs.qml_config import CFG, EnvConfig, QuantumConfig
from agent.quantum_actor import QuantumActor
from training.tmrl_integration import MockTMEnv


# ─────────────────────────────────────────────────────────────────────────────
# Classical baseline actor (standard MLP, same interface as QuantumActor)
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalActor(nn.Module):
    """Standard MLP actor (TMRL default architecture) for comparison."""

    def __init__(self, obs_dim: int = 22, act_dim: int = 3, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 2 * act_dim),
        )

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        out = self.net(obs)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        if deterministic:
            return torch.tanh(mean), None, torch.tanh(mean)
        from torch.distributions import Normal
        dist = Normal(mean, log_std.exp())
        x = dist.rsample()
        return torch.tanh(x), None, torch.tanh(mean)

    def num_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total}


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runners
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_latency(actor, obs_dim: int, n_warmup: int = 5, n_trials: int = 20) -> float:
    """Measure average inference latency in milliseconds."""
    dummy_obs = torch.rand(1, obs_dim)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            actor.get_action(dummy_obs, deterministic=True)

    # Timed trials
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        with torch.no_grad():
            actor.get_action(dummy_obs, deterministic=True)
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times))


def benchmark_rewards(actor, env: MockTMEnv, n_episodes: int = 10) -> List[float]:
    """Run N episodes and collect total rewards (random env, so relative comparison only)."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = actor.get_action(obs_t, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).numpy())
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark():
    print("\n" + "=" * 70)
    print("  QML-Trackmania: Classical vs Quantum Actor Benchmark")
    print("=" * 70)

    env_cfg = CFG.env
    q_cfg   = CFG.quantum

    # Create actors
    print("\nBuilding Classical Actor...")
    classical = ClassicalActor(env_cfg.obs_dim, env_cfg.act_dim)

    print("Building Quantum Actor...")
    quantum = QuantumActor(env_cfg, q_cfg)

    # ── Circuit stats ────────────────────────────────────────────────────────
    from quantum.circuits import build_full_circuit, print_circuit_summary
    qc = build_full_circuit(q_cfg.n_qubits, q_cfg.n_reps, q_cfg.entanglement)
    print("\n── Quantum Circuit ─────────────────────────────────────────────")
    print_circuit_summary(qc)

    # ── Parameter counts ─────────────────────────────────────────────────────
    print("\n── Parameter Counts ────────────────────────────────────────────")
    print(f"Classical Actor: {classical.num_parameters()['total']:,} params")
    q_params = quantum.num_parameters()
    for k, v in q_params.items():
        print(f"  Quantum Actor [{k}]: {v:,} params")

    # ── Latency benchmark ────────────────────────────────────────────────────
    print("\n── Inference Latency ───────────────────────────────────────────")
    classical_ms = benchmark_latency(classical, env_cfg.obs_dim)
    print(f"Classical: {classical_ms:.2f} ms/action")
    quantum_ms = benchmark_latency(quantum, env_cfg.obs_dim, n_warmup=2, n_trials=5)
    print(f"Quantum:   {quantum_ms:.2f} ms/action")
    print(f"Overhead:  {quantum_ms / classical_ms:.1f}×  (expected; circuit simulation cost)")

    # ── Episode reward benchmark ─────────────────────────────────────────────
    print("\n── Episode Rewards (MockTMEnv, untrained) ──────────────────────")
    env = MockTMEnv()
    c_rewards = benchmark_rewards(classical, env, n_episodes=5)
    q_rewards = benchmark_rewards(quantum,   env, n_episodes=5)
    print(f"Classical: mean={np.mean(c_rewards):.1f} ± {np.std(c_rewards):.1f}")
    print(f"Quantum:   mean={np.mean(q_rewards):.1f} ± {np.std(q_rewards):.1f}")
    print("(Untrained agents — rewards not meaningful; used only for pipeline check)")

    print("\n" + "=" * 70)
    print("Benchmark complete. Run quantum_sac.py to start actual training.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_benchmark()