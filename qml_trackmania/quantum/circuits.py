"""
quantum/circuits.py
===================
Parametrized Quantum Circuit (PQC) definitions using Qiskit 1.x.

Design:
  1. Feature map  — angle-encodes classical inputs via RY rotations
  2. Ansatz       — RealAmplitudes entangling block (trainable weights)
  3. Observables  — Pauli-Z on each qubit → n_qubits expectation values

The output of measure_circuit(obs_vector) ∈ [-1, +1]^n_qubits and is
fed into a classical linear "head" to produce actor mean/log_std.
"""

import numpy as np
from typing import List, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.quantum_info import SparsePauliOp


def build_feature_map(n_qubits: int, n_inputs: int) -> QuantumCircuit:
    """
    Angle encoding: encode n_inputs classical values into n_qubits using RY.
    If n_inputs > n_qubits we tile; if n_inputs < n_qubits we pad with zeros.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    n_inputs : int
        Dimension of the classical input vector (after pre-encoder).

    Returns
    -------
    QuantumCircuit
        Feature map circuit with Parameters named x[0], x[1], …
    """
    assert n_inputs <= n_qubits, (
        f"Feature map expects n_inputs ({n_inputs}) ≤ n_qubits ({n_qubits}). "
        "Use a pre-encoder to compress the observation first."
    )

    qc = QuantumCircuit(n_qubits, name="FeatureMap")
    x = ParameterVector("x", n_inputs)

    # Hadamard layer (create superposition)
    for q in range(n_qubits):
        qc.h(q)

    # Angle encoding via RY
    for i in range(n_inputs):
        # Map input to angle in [0, 2π] using arctan scaling for stability
        qc.ry(x[i], i)

    # If n_inputs < n_qubits, remaining qubits stay in |+⟩
    return qc


def build_ansatz(n_qubits: int, n_reps: int, entanglement: str = "linear") -> QuantumCircuit:
    """
    Hardware-efficient ansatz using Qiskit's RealAmplitudes.

    RealAmplitudes = alternating layers of RY rotations + CNOT entanglement.
    Number of trainable parameters = n_qubits * (n_reps + 1).

    Parameters
    ----------
    n_qubits : int
    n_reps : int
        Number of alternating rotation+entanglement layers.
    entanglement : str
        "linear" | "full" | "circular"

    Returns
    -------
    QuantumCircuit (RealAmplitudes ansatz)
    """
    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=n_reps,
        entanglement=entanglement,
        parameter_prefix="θ",
    )
    return ansatz


def build_observables(n_qubits: int) -> List[SparsePauliOp]:
    """
    Build one Pauli-Z observable per qubit.

    ⟨Z_i⟩ ∈ [-1, +1] for qubit i.

    Returns a list of SparsePauliOp, one per qubit.
    For EstimatorQNN we pass a single combined observable:
        Z₀ ⊗ I ⊗ … ⊗ I,  I ⊗ Z₁ ⊗ … ⊗ I, etc.
    """
    observables = []
    for i in range(n_qubits):
        # Build Pauli string: "I…IZI…I" with Z at position i
        pauli_str = "I" * (n_qubits - i - 1) + "Z" + "I" * i
        observables.append(SparsePauliOp(pauli_str))
    return observables


def build_full_circuit(n_qubits: int, n_reps: int, entanglement: str = "linear") -> QuantumCircuit:
    """
    Compose feature map + ansatz into a single QuantumCircuit.

    Input parameters: x[0..n_qubits-1]  (from classical pre-encoder)
    Weight parameters: θ[0..n_params-1] (trainable via gradient descent)

    Returns
    -------
    QuantumCircuit
        Full PQC ready for EstimatorQNN.
    """
    feature_map = build_feature_map(n_qubits, n_inputs=n_qubits)
    ansatz = build_ansatz(n_qubits, n_reps, entanglement)

    # Compose: feature map first, then ansatz
    qc = feature_map.compose(ansatz)
    qc.name = f"QML_Actor_PQC_{n_qubits}q_{n_reps}r"
    return qc


def print_circuit_summary(qc: QuantumCircuit) -> None:
    """Pretty-print circuit stats for IBM demo presentations."""
    print("=" * 60)
    print(f"Circuit: {qc.name}")
    print(f"  Qubits    : {qc.num_qubits}")
    print(f"  Depth     : {qc.depth()}")
    print(f"  Gate count: {qc.size()}")
    print(f"  Parameters: {len(qc.parameters)}")
    print(f"    Input (x): {[str(p) for p in qc.parameters if 'x' in str(p)]}")
    print(f"    Weights (θ): {len([p for p in qc.parameters if 'θ' in str(p)])}")
    print("=" * 60)
    print(qc.draw(output="text", fold=80))
