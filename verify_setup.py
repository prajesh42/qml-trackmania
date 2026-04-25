"""
verify_setup.py
===============
Run this first to verify your entire QML-Trackmania stack is working.

Usage:
    python verify_setup.py

Checks:
  1. Qiskit & Qiskit Aer import
  2. Qiskit Machine Learning (EstimatorQNN, TorchConnector)
  3. PyTorch
  4. Circuit build & draw
  5. QuantumLayer forward pass
  6. QuantumActor full forward pass with gradients
  7. TwinCritic
  8. ReplayBuffer sample
  9. MockTMEnv step
  10. One QML-SAC update step
"""

import sys, os

# Root of the project (ibm-trackmania/)
_ROOT = os.path.dirname(os.path.abspath(__file__))
# qml_trackmania/ package lives here — add it so `from quantum.x import y` works
_QML  = os.path.join(_ROOT, "qml_trackmania")

sys.path.insert(0, _ROOT)   # for verify_setup.py itself
sys.path.insert(0, _QML)    # so `quantum`, `agent`, `training`, etc. are importable

def check(name, fn):
    try:
        fn()
        print(f"  ✓  {name}")
    except Exception as e:
        print(f"  ✗  {name}: {e}")
        raise


def main():
    print("\n" + "=" * 60)
    print("  QML-Trackmania Setup Verification")
    print("=" * 60 + "\n")

    # 1. Core imports
    check("Qiskit import", lambda: __import__("qiskit"))
    check("Qiskit Aer import", lambda: __import__("qiskit_aer"))
    check("Qiskit ML import", lambda: __import__("qiskit_machine_learning"))
    check("PyTorch import", lambda: __import__("torch"))
    check("NumPy import", lambda: __import__("numpy"))

    # 2. Qiskit version check
    import qiskit, qiskit_aer, qiskit_machine_learning
    print(f"\n  Qiskit version:        {qiskit.__version__}")
    print(f"  Qiskit Aer version:    {qiskit_aer.__version__}")
    print(f"  Qiskit ML version:     {qiskit_machine_learning.__version__}")

    import torch
    print(f"  PyTorch version:       {torch.__version__}")
    print()

    # 3. Circuit build
    from quantum.circuits import build_full_circuit
    check("Circuit build", lambda: build_full_circuit(4, 2))

    # 4. QuantumLayer
    from configs.qml_config import QuantumConfig
    from quantum.qnn import QuantumLayer
    q_cfg = QuantumConfig(n_qubits=4, n_reps=2)
    ql = None
    def _build_ql():
        nonlocal ql
        ql = QuantumLayer(q_cfg)
    check("QuantumLayer build", _build_ql)

    # 5. QuantumLayer forward
    import torch
    x = torch.rand(2, 4)
    check("QuantumLayer forward", lambda: ql(x))

    # 6. QuantumActor
    from configs.qml_config import EnvConfig
    from agent.quantum_actor import QuantumActor
    env_cfg = EnvConfig()
    actor = None
    def _build_actor():
        nonlocal actor
        actor = QuantumActor(env_cfg, q_cfg)
    check("QuantumActor build", _build_actor)

    obs = torch.rand(2, env_cfg.obs_dim)
    check("QuantumActor forward", lambda: actor.get_action(obs))

    # 7. Gradient check
    def _grad_check():
        action, log_prob, _ = actor.get_action(obs)
        loss = -log_prob.mean()
        loss.backward()
    check("QuantumActor backward (gradients)", _grad_check)

    # 8. TwinCritic
    from configs.qml_config import CriticConfig
    from agent.classical_critic import TwinCritic
    critic_cfg = CriticConfig()
    critic = TwinCritic(env_cfg, critic_cfg)
    check("TwinCritic forward",
          lambda: critic(obs, torch.rand(2, env_cfg.act_dim)))

    # 9. ReplayBuffer
    from agent.replay_buffer import ReplayBuffer
    buf = ReplayBuffer(env_cfg.obs_dim, env_cfg.act_dim, capacity=1000)
    import numpy as np
    for _ in range(300):
        buf.add(
            np.random.rand(env_cfg.obs_dim).astype(np.float32),
            np.random.rand(env_cfg.act_dim).astype(np.float32),
            float(np.random.rand()),
            np.random.rand(env_cfg.obs_dim).astype(np.float32),
            False,
        )
    check("ReplayBuffer sample", lambda: buf.sample(64))

    # 10. MockTMEnv
    from training.tmrl_integration import MockTMEnv
    env = MockTMEnv()
    o, _ = env.reset()
    check("MockTMEnv step", lambda: env.step(env.action_space.sample()))

    # 11. One SAC update
    from configs.qml_config import CFG
    from training.quantum_sac import QuantumSAC
    CFG.sac.batch_size = 64
    agent = QuantumSAC(CFG, device="cpu")
    # Pre-fill buffer
    for _ in range(64):
        agent.buffer.add(
            np.random.rand(env_cfg.obs_dim).astype(np.float32),
            np.random.rand(env_cfg.act_dim).astype(np.float32),
            float(np.random.rand()),
            np.random.rand(env_cfg.obs_dim).astype(np.float32),
            False,
        )
    check("QML-SAC update step", lambda: agent.update())

    print("\n" + "=" * 60)
    print("  All checks passed! 🎉")
    print("  Run: python training/quantum_sac.py  to start training")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()