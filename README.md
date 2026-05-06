# qml-trackmania

Quantum-enhanced Trackmania training stack built on `tmrl`, with a custom `QSAC` pipeline powered by Qiskit Aer.

## What This Repository Provides

- Source-level `tmrl` customization for Trackmania RL workflows.
- `QSAC` algorithm integration through `ALG.ALGORITHM = "QSAC"`.
- Quantum actor-critic implementation in `tmrl/custom/quantum/quantum_models.py`.
- Aer simulator device diagnostics and selection (`AUTO`, `CPU`, `GPU`).
- Worker-side control stabilization for more natural early driving behavior.

## Architecture

`QSAC` keeps the SAC training logic and injects Quantum layers directly into the policy/value function stack used for action selection and reward-driven learning.

- Actor path: `state -> Quantum feature extractor -> action distribution -> action`.
- Critic path: `(state, action) -> Quantum feature extractor -> Q-value`.
- Reward path: `environment -> reward` supervises actor/critic updates through SAC losses.

## Quantum in RL Action-Reward Loop

```text
[Current state from game]
          |
          v
+-------------------------------+
| Quantum RL Model              |
| - Quantum Policy (choose move)|
| - Quantum Value (judge move)  |
+-------------------------------+
          |
          v
   [Action in Trackmania]
          |
          v
   [New state + Reward]
          |
          v
[Training update from reward]
          |
          +---------------------> back to Quantum RL Model
```

## Requirements

- Python `3.10` to `3.12` recommended.
- Trackmania 2020 and OpenPlanet plugin configured.
- Initialized `TmrlData` directory.
- For Windows local worker: `pywin32` and gamepad dependencies.

## Installation

Use the same environment for install and execution.

### 1. Create virtual environment

Windows PowerShell:

```powershell
python -m venv tmrl_env
.\tmrl_env\Scripts\Activate.ps1
```

Linux:

```bash
python -m venv tmrl_env
source tmrl_env/bin/activate
```

### 2. Install from source (editable)

From repository root:

```bash
python -m pip install -U pip
python -m pip install -e .[qiskit]
```

Fallback if extras are unavailable:

```bash
python -m pip install -e .
python -m pip install qiskit-aer
```

### 3. Validate environment path

```bash
python - <<'PY'
import tmrl
print(tmrl.__file__)
PY
```

Expected result: path points to your active environment with your current source install, not a stale global install.

## Configuration

Primary config file:

- Windows: `C:\Users\<user>\TmrlData\config\config.json`
- Linux: `~/TmrlData/config/config.json`

### Minimal QSAC block

```json
"ALG": {
  "ALGORITHM": "QSAC",
  "QUANTUM_N_QUBITS": 3,
  "QUANTUM_N_LAYERS": 1,
  "QUANTUM_ENCODER_SEED": 0,
  "QUANTUM_AER_DEVICE": "AUTO"
}
```

### Worker stability profile for QSAC

```json
"RTGYM_CONFIG": {
  "time_step_duration": 0.10,
  "start_obs_capture": 0.08,
  "time_step_timeout_factor": 2.5,
  "act_buf_len": 2,
  "benchmark": false,
  "wait_on_done": true
}
```

## Operational Runbook

### 1. Record reward trajectory

```bash
python -m tmrl --record-reward
```

### 2. Validate environment and reward

```bash
python -m tmrl --check-environment
```

### 3. Start training services

Terminal A:

```bash
python -m tmrl --server
```

Terminal B:

```bash
python -m tmrl --trainer
```

Terminal C:

```bash
python -m tmrl --worker
```

## Aer Backend Modes

Current `QSAC` implementation uses `qiskit_aer.AerSimulator` with `statevector`.

Device selection is controlled by:

```json
"QUANTUM_AER_DEVICE": "AUTO"
```

Valid values:

- `"AUTO"`: choose GPU when available in Aer, else CPU.
- `"GPU"`: force GPU, fallback warning if unsupported.
- `"CPU"`: force CPU.

### Important platform note

GPU Aer wheels are generally available for Linux x86_64 environments.  
If Aer reports `('CPU',)`, your current Aer build in that environment is CPU-only.

## Performance Guidance

- Quantum parameter count scales as `3 * n_qubits * n_layers`.
- Statevector simulation cost rises quickly with qubit count.
- Keep `QUANTUM_N_LAYERS=1` initially.
- Start with `QUANTUM_N_QUBITS` in `2..3`.
- Increase worker time-step duration before increasing model complexity.

## Observability and Verification

### Verify Torch and Aer capabilities

```bash
python - <<'PY'
import torch
from qiskit_aer import AerSimulator
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("torch GPU:", torch.cuda.get_device_name(0))
sim = AerSimulator(method="statevector")
print("Aer available_devices:", sim.available_devices())
print("Aer device option:", sim.options.device)
PY
```

### Verify active algorithm in config

```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path.home() / "TmrlData" / "config" / "config.json"
cfg = json.loads(p.read_text(encoding="utf-8-sig"))
print(cfg["ALG"]["ALGORITHM"])
PY
```

## Troubleshooting

### `AssertionError: ... implement QSAC ...`

Cause: old installed `tmrl` package without QSAC support.  
Fix:

```bash
python -m pip uninstall -y tmrl
python -m pip install -e .[qiskit]
```

### `ImportError: QSAC requires qiskit-aer`

Cause: Aer not installed in trainer environment.  
Fix:

```bash
python -m pip install qiskit-aer
```

### `ImportError: cannot import name 'convert_to_target' ...`

Cause: incompatible `qiskit` and `qiskit-aer*` versions.  
Fix: reinstall compatible versions in the same active environment.

### Many worker timestep timeouts

Cause: rollout loop too tight for current environment/model speed.  
Fix:

- increase `time_step_duration`
- increase `time_step_timeout_factor`
- reduce `QUANTUM_N_QUBITS`
- keep `QUANTUM_N_LAYERS` low

### Car brakes or reverses too much at startup

This repo includes:

- control signal sanitization in TM interface
- forward-biased initialization in quantum actor priors

Restart trainer and worker after pulling latest changes.

## Repository Layout

- `tmrl/custom/quantum/`: QSAC quantum model code.
- `tmrl/custom/custom_algorithms.py`: SAC/REDQ agents and trainer-device logs.
- `tmrl/custom/tm/`: Trackmania environment interfaces and controls.
- `readme/get_started.md`: detailed walkthrough.
- `readme/reference_guide.md`: config key reference.
- `tmrl_readme.md`: upstream-style broad TMRL documentation snapshot.
