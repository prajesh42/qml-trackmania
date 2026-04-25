# QML-Trackmania: Quantum Machine Learning for Autonomous Racing

> **IBM × Trackmania Demo** — Hybrid Quantum-Classical SAC agent using Qiskit Aer + TMRL

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        TMRL Environment                         │
│   TrackMania 2020  ←→  Gymnasium Interface (LIDAR obs space)   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ obs: [lidar×19, speed, gear, rpm]
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Quantum SAC Agent                             │
│                                                                 │
│  ┌──────────────────────┐    ┌───────────────────────────────┐  │
│  │   Quantum Actor      │    │   Classical Twin Critics      │  │
│  │                      │    │                               │  │
│  │  Angle Encoding      │    │   MLP(obs+action) → Q-value  │  │
│  │  ↓                   │    │   MLP(obs+action) → Q-value  │  │
│  │  PQC (4 qubits)      │    │   (min of two for stability) │  │
│  │  [RY + CNOT ansatz]  │    └───────────────────────────────┘  │
│  │  ↓                   │                                       │
│  │  Pauli-Z measure     │    ┌───────────────────────────────┐  │
│  │  ↓                   │    │   Replay Buffer (off-policy)  │  │
│  │  Classical head      │    │   capacity: 500,000           │  │
│  │  → mean, log_std     │    └───────────────────────────────┘  │
│  └──────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
                            │ actions: [gas, brake, steering]
                            ▼
                    TrackMania 2020
```

## Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Quantum simulator | Qiskit Aer (StatevectorSimulator) | IBM stack, noiseless for demo |
| QNN type | EstimatorQNN (Pauli-Z observables) | Continuous output → actor policy |
| QNN integration | TorchConnector | Auto-differentiation through PQC |
| Encoding | Angle encoding (RY gates) | Scales to NISQ hardware naturally |
| Ansatz | RealAmplitudes (2 reps) | Expressible, hardware-efficient |
| Actor | Quantum (4 qubits) | Demo showcase of QML |
| Critics | Classical MLP | Training stability |
| Algorithm | SAC (soft actor-critic) | Matches TMRL default pipeline |

---

## Project Structure

```
qml_trackmania/
├── README.md                    ← This file
├── requirements.txt             ← All Python dependencies
├── configs/
│   └── qml_config.py           ← Hyperparameters & circuit config
├── quantum/
│   ├── circuits.py             ← PQC circuit definitions
│   └── qnn.py                  ← EstimatorQNN + TorchConnector setup
├── agent/
│   ├── quantum_actor.py        ← Hybrid quantum-classical actor network
│   ├── classical_critic.py     ← Twin-Q classical critic networks
│   └── replay_buffer.py        ← Off-policy experience replay
├── training/
│   ├── quantum_sac.py          ← Main QML-SAC training loop
│   └── tmrl_integration.py     ← TMRL environment wrapper
├── utils/
│   ├── logger.py               ← W&B / CSV training logger
│   └── benchmark.py            ← Classical vs Quantum comparison
└── notebooks/
    └── qml_demo.ipynb          ← IBM demo notebook
```

---

## Quick Start

### Step 1 — Prerequisites

You need:
- **TrackMania 2020** (free on Ubisoft Connect)
- **OpenPlanet** mod loader (for the TMRL plugin)
- **TMRL** already installed and running (as stated: ✅ done)
- Python ≥ 3.10

### Step 2 — Install QML Dependencies

```bash
# In your TMRL virtual environment (or a new one)
pip install qiskit==1.3.2
pip install qiskit-aer==0.15.1
pip install qiskit-machine-learning==0.8.2   # TorchConnector + EstimatorQNN
pip install torch>=2.1.0
pip install numpy scipy matplotlib wandb
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Step 3 — Verify Qiskit Aer Works

```python
from qiskit_aer import AerSimulator
sim = AerSimulator(method='statevector')
print("Qiskit Aer OK:", sim.name)
```

### Step 4 — Configure

Edit `configs/qml_config.py` to match your machine. Key knobs:

```python
N_QUBITS = 4          # increase for richer policy, but slower
N_REPS   = 2          # ansatz repetitions
OBS_DIM  = 19 + 3     # 19 LIDAR rays + speed + gear + rpm  (TMRL default)
ACT_DIM  = 3          # [gas, brake, steering]
```

### Step 5 — Run Training

```bash
# Terminal 1: Start TMRL server (as normal)
python -m tmrl --server

# Terminal 2: Start TMRL worker (game interface)
python -m tmrl --worker

# Terminal 3: Start QML trainer (replaces default tmrl --trainer)
python training/quantum_sac.py
```

---

## How Quantum Replaces Classical

In standard TMRL-SAC the actor is a **Multi-Layer Perceptron**:
```
obs → Linear(22, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 6) → [mean, log_std]
```

In QML-SAC the actor becomes a **Hybrid Quantum-Classical network**:
```
obs → Classical pre-encoder (22→4) → Angle Encoding (RY gates)
    → PQC Ansatz (RealAmplitudes, 2 reps, 4 qubits)
    → Pauli-Z measurement (4 expectation values)
    → Classical post-head (4→6) → [mean, log_std for 3 actions]
```

The Qiskit `TorchConnector` makes the quantum layer a drop-in PyTorch module, so the rest of SAC (critics, replay buffer, entropy tuning) is unchanged.

---

## References

- TMRL: https://github.com/trackmania-rl/tmrl
- Qiskit Machine Learning: https://github.com/qiskit-community/qiskit-machine-learning  
- Quantum SAC paper: https://arxiv.org/abs/2401.07043
- RealAmplitudes ansatz: Qiskit circuit library
- Parameter Shift Rule (gradients): Schuld et al. 2019