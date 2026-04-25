"""
training/tmrl_integration.py
============================
Integration layer between the QML-SAC agent and the TMRL framework.

TMRL provides a Gymnasium environment for TrackMania 2020.
This module handles:
  1. Creating the TMRL Gymnasium env (LIDAR mode)
  2. Normalising observations to [-1, 1] range (important for angle encoding)
  3. Flattening/reshaping observation for the quantum actor
  4. Providing a standalone test mode (no game needed)

TMRL Environment Observation Space (LIDAR mode)
────────────────────────────────────────────────
  obs = (lidar[0..18], speed, gear, rpm)
        ├── lidar: 19 ray distances, normalised ∈ [0, 1]
        ├── speed: m/s, raw (≈ 0–300)
        ├── gear: 1–6
        └── rpm: engine RPM (≈ 0–10000)

We normalise speed and rpm to [0, 1] for stable encoding.

TMRL Action Space
─────────────────
  action = [gas, brake, steering]  ∈ [-1, 1]^3
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple
import logging

log = logging.getLogger(__name__)

# Normalisation constants (from empirical TMRL ranges)
SPEED_MAX = 300.0     # km/h equivalent
GEAR_MAX  = 6.0
RPM_MAX   = 10_000.0


def normalise_obs(obs: np.ndarray) -> np.ndarray:
    """
    Normalise raw TMRL observation to [-1, 1] for quantum angle encoding.

    Structure: [lidar×19, speed, gear, rpm]
    - lidar values are already in [0, 1] from TMRL → rescale to [-1, 1]
    - speed, gear, rpm are raw → normalise then rescale

    Returns float32 array of shape (22,).
    """
    obs = obs.astype(np.float32).flatten()

    if len(obs) >= 22:
        # LIDAR: [0,1] → [-1, 1]
        obs[:19] = obs[:19] * 2.0 - 1.0
        # Speed
        obs[19] = obs[19] / SPEED_MAX * 2.0 - 1.0
        # Gear: 1–6 → [0,1] → [-1,1]
        obs[20] = obs[20] / GEAR_MAX * 2.0 - 1.0
        # RPM
        obs[21] = obs[21] / RPM_MAX * 2.0 - 1.0

    return np.clip(obs, -1.0, 1.0)


class TMRLWrapper(gym.Wrapper):
    """
    Wraps the TMRL Gymnasium environment with observation normalisation.

    This ensures the quantum actor always receives inputs ∈ [-1, 1],
    which map cleanly to rotation angles ∈ [-π, π] via the pre-encoder.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        return normalise_obs(obs), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return normalise_obs(obs), reward, terminated, truncated, info


def make_tmrl_env(cfg=None) -> gym.Env:
    """
    Create and return the TMRL TrackMania Gymnasium environment.

    Falls back to a mock environment if TMRL server is not reachable
    (useful for unit testing and circuit development).

    Parameters
    ----------
    cfg : Config, optional

    Returns
    -------
    gym.Env
    """
    try:
        import tmrl.config.config_constants as cfg_const
        from tmrl.training_offline import TmrlData
        import gymnasium

        # Use the standard TMRL LIDAR environment
        # This requires the TMRL server to be running
        env = gymnasium.make("real-time-gym-v1",
                             interface="TM20LidarInterface",
                             time_step_duration=0.05,
                             start_obs_capture=0.04,
                             time_step_timeout_factor=1.0,
                             ep_max_length=1000,
                             act_buf_len=1,
                             )
        env = TMRLWrapper(env)
        log.info("TMRL TrackMania environment created ✓")
        return env

    except Exception as e:
        log.warning(f"Could not connect to TMRL server: {e}")
        log.warning("Falling back to MockTMEnv for offline development.")
        return MockTMEnv()


class MockTMEnv(gym.Env):
    """
    Synthetic TrackMania environment for testing without the game.

    Mimics the TMRL LIDAR observation/action spaces so all code
    (quantum circuits, SAC updates) can be developed and tested offline.

    Observation: 22-dim float32 ∈ [-1, 1]
    Action:       3-dim float32 ∈ [-1, 1]
    Reward:       random ∈ [0, 1] (placeholder)
    """

    def __init__(self, obs_dim: int = 22, act_dim: int = 3, max_steps: int = 500):
        super().__init__()
        self.obs_dim   = obs_dim
        self.act_dim   = act_dim
        self.max_steps = max_steps
        self._step     = 0

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step = 0
        obs = self.observation_space.sample()
        return obs.astype(np.float32), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._step += 1
        obs = self.observation_space.sample()
        # Reward: proportional to forward speed (LIDAR ray 9 = centre-forward)
        reward = float(np.clip(obs[9], 0, 1))  # placeholder
        terminated = False
        truncated  = self._step >= self.max_steps
        return obs.astype(np.float32), reward, terminated, truncated, {}

    def render(self): pass
    def close(self): pass
