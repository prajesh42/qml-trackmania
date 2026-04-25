"""
quantum/qnn.py
==============
Wraps the PQC as a Qiskit EstimatorQNN and exposes it as a PyTorch module
via TorchConnector.

The resulting `QuantumLayer` is a standard nn.Module:
    forward(x: Tensor[batch, n_qubits]) → Tensor[batch, n_qubits]

All gradients are computed via the Parameter-Shift Rule (exact) or SPSA
(approximate, faster) depending on config.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from quantum.circuits import build_full_circuit, build_observables
from configs.qml_config import QuantumConfig


def resolve_aer_backend_options(cfg: QuantumConfig) -> tuple[dict, str]:
    """
    Choose Aer backend options, preferring GPU when available and requested.
    """
    probe = AerSimulator()
    available_devices = {device.upper() for device in probe.available_devices()}

    requested_device = cfg.aer_device.upper()
    if requested_device == "AUTO":
        device = "GPU" if "GPU" in available_devices else "CPU"
    elif requested_device in available_devices:
        device = requested_device
    else:
        device = "CPU"

    options = {"method": cfg.aer_method, "device": device}
    return options, device


def build_estimator(cfg: QuantumConfig) -> tuple[AerEstimator, AerSimulator, str]:
    """
    Create an Aer EstimatorV2 primitive backed by the statevector simulator.

    For noisy simulation add:
        from qiskit_aer.noise import NoiseModel
        backend = AerSimulator.from_backend(real_backend)

    For real IBM hardware swap AerEstimator with:
        from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = service.least_busy(...)
        estimator = EstimatorV2(backend)
    """
    backend_options, device = resolve_aer_backend_options(cfg)
    backend = AerSimulator(**backend_options)
    estimator = AerEstimator(options={"backend_options": backend_options})
    return estimator, backend, device


def build_qnn(cfg: QuantumConfig) -> EstimatorQNN:
    """
    Assemble the EstimatorQNN from:
      - PQC (feature map + ansatz)
      - Pauli-Z observables (one per qubit)
      - Aer estimator backend

    Returns
    -------
    EstimatorQNN
        QNN with:
          input_params  = x[0..n_qubits-1]
          weight_params = θ[0..n_params-1]
          output_shape  = (n_qubits,)   one ⟨Z_i⟩ per qubit
    """
    qc = build_full_circuit(cfg.n_qubits, cfg.n_reps, cfg.entanglement)
    observables = build_observables(cfg.n_qubits)
    estimator, backend, aer_device = build_estimator(cfg)

    # Aer cannot execute high-level library instructions like RealAmplitudes
    # directly in every version combo, so compile the PQC to basis gates first.
    qc = transpile(qc.decompose(), backend=backend, optimization_level=0)

    # Separate input params (x) from weight params (θ)
    input_params = sorted(
        [p for p in qc.parameters if "x" in str(p)],
        key=lambda p: str(p)
    )
    weight_params = sorted(
        [p for p in qc.parameters if "θ" in str(p)],
        key=lambda p: str(p)
    )

    qnn = EstimatorQNN(
        circuit=qc,
        observables=observables,           # list of SparsePauliOp
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
        input_gradients=False,             # we only need weight gradients
    )

    print(f"[QNN] Built EstimatorQNN — "
          f"qubits={cfg.n_qubits}, reps={cfg.n_reps}, "
          f"params={len(weight_params)}, output_dim={cfg.n_qubits}, "
          f"aer_method={cfg.aer_method}, aer_device={aer_device}")
    return qnn


class QuantumLayer(nn.Module):
    """
    PyTorch nn.Module wrapping a Qiskit EstimatorQNN via TorchConnector.

    This layer is fully differentiable: PyTorch backprop flows through the
    TorchConnector which calls Qiskit's parameter-shift gradient internally.

    Parameters
    ----------
    cfg : QuantumConfig
    initial_weights : np.ndarray, optional
        Warm-start weights (e.g., loaded from checkpoint).

    Input  shape: (batch, n_qubits)   — encoded observation
    Output shape: (batch, n_qubits)   — Pauli-Z expectation values ∈ [-1,1]
    """

    def __init__(self, cfg: QuantumConfig, initial_weights: Optional[np.ndarray] = None):
        super().__init__()
        self.cfg = cfg

        # Build Qiskit QNN
        self._qnn = build_qnn(cfg)

        # Initialise weights
        if initial_weights is None:
            rng = np.random.default_rng(42)
            initial_weights = rng.uniform(
                -np.pi, np.pi, self._qnn.num_weights
            )

        # TorchConnector wraps QNN as nn.Module
        # initial_weights must be a 1-D float Tensor
        self._connector = TorchConnector(
            self._qnn,
            initial_weights=torch.tensor(initial_weights, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, n_qubits)
            Angle-encoded observation (output of pre-encoder, after arctan scaling).

        Returns
        -------
        Tensor of shape (batch, n_qubits)
            ⟨Z_i⟩ expectation values, each ∈ [-1, 1].
        """
        # TorchConnector processes one sample at a time when batched
        # We loop over batch dim for compatibility (Qiskit primitives handle batch natively in V2)
        return self._connector(x)

    @property
    def quantum_weights(self) -> torch.Tensor:
        """Return the trainable PQC weights as a flat tensor."""
        return self._connector.weight

    @property
    def num_params(self) -> int:
        return self._qnn.num_weights


def test_quantum_layer():
    """Smoke test — run this to verify the quantum stack works."""
    from configs.qml_config import QuantumConfig
    cfg = QuantumConfig(n_qubits=4, n_reps=2)

    print("Building QuantumLayer...")
    ql = QuantumLayer(cfg)
    print(f"  Parameters: {ql.num_params}")

    # Fake batch of 3 encoded observations
    x = torch.rand(3, cfg.n_qubits, requires_grad=False)
    print(f"  Input shape: {x.shape}")

    out = ql(x)
    print(f"  Output shape: {out.shape}")
    print(f"  Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print("QuantumLayer OK ✓")


if __name__ == "__main__":
    test_quantum_layer()
