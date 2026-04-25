"""
configs/qml_config.py
=====================
Central configuration for the QML-Trackmania project.
All hyperparameters live here — edit this file to tune the demo.
"""

from dataclasses import dataclass, field
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# TMRL / Environment
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EnvConfig:
    """
    Matches the default TMRL LIDAR environment.
    LIDAR: 19 rays + speed (m/s) + gear (int) + rpm → 22-dim obs.
    Actions: [gas ∈ [-1,1], brake ∈ [-1,1], steering ∈ [-1,1]].
    """
    obs_dim: int = 22           # 19 LIDAR + speed + gear + rpm
    act_dim: int = 3            # gas, brake, steering
    act_low: float = -1.0
    act_high: float = 1.0

    # TMRL server address (localhost for single-machine setup)
    server_ip: str = "127.0.0.1"
    server_port: int = 55555

    # Observation history (TMRL stacks N frames by default)
    obs_history: int = 1        # set to 4 to enable frame stacking


# ─────────────────────────────────────────────────────────────────────────────
# Quantum Circuit
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class QuantumConfig:
    """
    PQC design choices.
    - n_qubits:  number of qubits (=quantum feature dim)
    - n_reps:    repetitions of the RealAmplitudes ansatz
    - encoding:  angle encoding via RY gates
    - backend:   'aer_simulator' for Qiskit Aer simulation,
                 or an IBM Quantum backend string for real hardware
    """
    n_qubits: int = 4               # 4 qubits → 2^4 = 16-dim Hilbert space
    n_reps: int = 1                 # reduced circuit depth (was 2)
    shots: int = 1024              # for SamplerQNN; ignored by EstimatorQNN
    entanglement: str = "linear"   # "linear" | "full" | "circular"
    backend: str = "aer_simulator"  # swap to "ibm_kyoto" etc. for hardware
    aer_method: str = "statevector" # "statevector", "density_matrix", etc.
    aer_device: str = "auto"        # "auto" | "CPU" | "GPU"

    # Pre-encoder: maps obs_dim → n_qubits  (classical linear layer)
    use_pre_encoder: bool = True
    pre_encoder_hidden: int = 16

    # Post-head: maps n_qubits → 2*act_dim  (μ and log_σ for each action)
    use_post_head: bool = True

    # Gradient method: "param_shift" | "spsa"
    # spsa is faster (2 evals per param); param_shift is exact but slower
    gradient_method: str = "spsa"

    @property
    def n_params(self) -> int:
        """Number of trainable parameters in RealAmplitudes ansatz."""
        # RealAmplitudes: n_qubits * (n_reps + 1) rotations
        return self.n_qubits * (self.n_reps + 1)


# ─────────────────────────────────────────────────────────────────────────────
# SAC Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SACConfig:
    """
    Soft Actor-Critic hyperparameters.
    Tuned conservatively for the quantum actor (slower convergence expected).
    """
    gamma: float = 0.99             # discount factor
    tau: float = 0.005              # soft target update rate
    alpha: float = 0.2              # initial entropy temperature
    auto_alpha: bool = False        # disable for speed (no entropy tuning)
    target_entropy: float = -3.0   # = -act_dim by default

    # Learning rates — lower for quantum actor (PQC gradients are noisier)
    actor_lr: float = 1e-3          # higher to compensate for less frequent updates
    critic_lr: float = 1e-3         # higher to compensate for less frequent updates
    alpha_lr: float = 3e-4

    # Replay buffer
    buffer_size: int = 500_000
    batch_size: int = 32            # very small for fast quantum evals

    # Training schedule
    warmup_steps: int = 1_000      # random actions before learning starts
    update_every: int = 200         # update every 200 env steps (collect data faster)
    updates_per_step: int = 4       # batch multiple updates together


# ─────────────────────────────────────────────────────────────────────────────
# Critic Network (classical MLP)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CriticConfig:
    hidden_sizes: Tuple[int, ...] = (256, 256)
    activation: str = "relu"


# ─────────────────────────────────────────────────────────────────────────────
# Training Run
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    run_name: str = "QML_SAC_trackmania"
    max_steps: int = 1_000_000
    eval_every: int = 10_000        # steps between evaluation episodes
    save_every: int = 50_000        # steps between checkpoint saves
    checkpoint_dir: str = "checkpoints"
    log_wandb: bool = False         # set True to enable Weights & Biases
    wandb_project: str = "qml-trackmania"
    seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Master Config (combine all above)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Singleton for easy import
CFG = Config()
