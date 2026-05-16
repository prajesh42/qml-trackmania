"""
Quantum-ready policy/value models for TrackMania LIDAR pipelines.

These models are "hybrid" and can run fully in torch out of the box.
The feature mapping uses phase/amplitude style trigonometric blocks for safe
fallback behavior, and optionally uses a real Qiskit statevector backend
behind a config flag.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from tmrl.actor import TorchActorModule
from tmrl.util import prod


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def _bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _obs_dim_and_mode(observation_space):
    try:
        dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
        tuple_obs = True
    except TypeError:
        dim_obs = prod(observation_space.shape)
        tuple_obs = False
    return dim_obs, tuple_obs


class HybridQuantumFeatureMap(nn.Module):
    """
    Lightweight trigonometric feature map.

    This is intentionally deterministic and torch-native for low-latency use in
    real-time control loops. It behaves like a quantum-inspired embedding layer.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, depth: int = 2):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be > 0, got {depth}")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.phase_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(depth))
        self.amp_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(depth))
        self.norm_layers = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(depth))

    def forward(self, x):
        x = self.input_proj(x)
        for phase, amp, norm in zip(self.phase_layers, self.amp_layers, self.norm_layers):
            update = torch.sin(phase(x)) * torch.cos(amp(x))
            x = norm(x + update)
        return x


class QiskitStatevectorFeatureMap(nn.Module):
    """
    True Qiskit backend using statevector simulation.

    This module has no trainable quantum parameters by design. It produces
    stable quantum features that are consumed by trainable torch heads.
    """
    def __init__(
        self,
        input_dim: int,
        num_qubits: int = 6,
        reuploads: int = 1,
        angle_scale: float = float(np.pi),
        seed: int = 1234,
    ):
        super().__init__()
        if num_qubits <= 0:
            raise ValueError(f"num_qubits must be > 0, got {num_qubits}")
        if reuploads <= 0:
            raise ValueError(f"reuploads must be > 0, got {reuploads}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")

        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
        except Exception as exc:
            raise ModuleNotFoundError(
                "Qiskit backend requested but qiskit is not importable. "
                "Install qiskit or disable the Qiskit backend."
            ) from exc

        self._QuantumCircuit = QuantumCircuit
        self._Statevector = Statevector

        self.input_dim = int(input_dim)
        self.num_qubits = int(num_qubits)
        self.reuploads = int(reuploads)
        self.angle_scale = float(angle_scale)
        self.seed = int(seed)

        gen = torch.Generator()
        gen.manual_seed(self.seed)

        base_projection = torch.randn(self.input_dim, self.num_qubits, generator=gen)
        base_projection = base_projection / np.sqrt(float(self.input_dim))
        self.register_buffer("base_projection", base_projection)

        if self.reuploads > 1:
            rep = torch.randn(self.reuploads - 1, self.input_dim, self.num_qubits, generator=gen)
            rep = rep / np.sqrt(float(self.input_dim))
        else:
            rep = torch.empty(0, self.input_dim, self.num_qubits)
        self.register_buffer("reupload_projections", rep)

        n_basis = 1 << self.num_qubits
        basis = np.arange(n_basis, dtype=np.int64)
        sign_rows = []
        for qubit in range(self.num_qubits):
            bits = ((basis >> qubit) & 1).astype(np.float32)
            sign_rows.append(1.0 - 2.0 * bits)
        sign_cache = torch.tensor(np.stack(sign_rows, axis=0), dtype=torch.float32)
        self.register_buffer("z_sign_cache", sign_cache, persistent=False)

    def _evaluate_single(self, layered_angles):
        qc = self._QuantumCircuit(self.num_qubits)
        for layer_idx, angles in enumerate(layered_angles):
            for qubit in range(self.num_qubits):
                qc.ry(float(angles[qubit]), qubit)
            for qubit in range(self.num_qubits - 1):
                qc.cx(qubit, qubit + 1)
            if self.num_qubits > 2 and (layer_idx % 2 == 0):
                qc.cx(self.num_qubits - 1, 0)
        state = self._Statevector.from_instruction(qc)
        probs = np.asarray(state.probabilities(), dtype=np.float32)
        signs = self.z_sign_cache.cpu().numpy()
        exp_z = signs @ probs
        return exp_z.astype(np.float32)

    def forward(self, x):
        if x.ndim != 2:
            x = torch.flatten(x, start_dim=1)
        x_cpu = x.detach().to(dtype=torch.float32, device="cpu")
        device = x.device
        dtype = x.dtype

        base_projection = self.base_projection.cpu()
        base_angles = torch.tanh(x_cpu @ base_projection) * self.angle_scale
        layered = [base_angles]

        rep_proj = self.reupload_projections.cpu()
        for idx in range(rep_proj.shape[0]):
            angles = torch.tanh(x_cpu @ rep_proj[idx]) * self.angle_scale
            layered.append(angles)

        outputs = []
        for sample_idx in range(x_cpu.shape[0]):
            sample_layers = [layer[sample_idx].numpy() for layer in layered]
            outputs.append(self._evaluate_single(sample_layers))

        out_np = np.stack(outputs, axis=0)
        out = torch.tensor(out_np, device=device, dtype=dtype)
        return out


class QuantumFeatureBackbone(nn.Module):
    """
    Backend switch with safe fallback behavior.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
        quantum_backend: str = "fallback",
        qiskit_num_qubits: int = 6,
        qiskit_reuploads: int = 1,
        qiskit_angle_scale: float = float(np.pi),
        qiskit_seed: int = 1234,
        qiskit_fallback_on_error: bool = True,
        qiskit_strict: bool = False,
        quantum_residual_gain: float = 0.05,
    ):
        super().__init__()
        self.fallback = HybridQuantumFeatureMap(input_dim=input_dim, hidden_dim=hidden_dim, depth=depth)
        self.quantum_backend = str(quantum_backend).strip().lower()
        self.qiskit_fallback_on_error = _bool(qiskit_fallback_on_error)
        self.qiskit_strict = _bool(qiskit_strict)
        self._qiskit_runtime_disabled = False

        if self.quantum_backend not in {"fallback", "qiskit"}:
            raise ValueError(f"Unsupported quantum_backend: {self.quantum_backend}")

        self.qiskit_map = None
        self.qiskit_head = None
        if self.quantum_backend == "qiskit":
            try:
                self.qiskit_map = QiskitStatevectorFeatureMap(
                    input_dim=input_dim,
                    num_qubits=qiskit_num_qubits,
                    reuploads=qiskit_reuploads,
                    angle_scale=qiskit_angle_scale,
                    seed=qiskit_seed,
                )
                self.qiskit_head = nn.Sequential(
                    nn.Linear(qiskit_num_qubits, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                logging.info(
                    f"Quantum backend enabled: qiskit (qubits={qiskit_num_qubits}, reuploads={qiskit_reuploads})"
                )
            except Exception as exc:
                if self.qiskit_strict:
                    raise RuntimeError("Qiskit backend initialization failed in strict mode.") from exc
                if not self.qiskit_fallback_on_error:
                    raise RuntimeError("Qiskit backend initialization failed and fallback is disabled.") from exc
                logging.warning(f"Qiskit backend init failed ({exc}). Falling back to torch quantum feature map.")
                self.qiskit_map = None
                self.qiskit_head = None
                self.quantum_backend = "fallback"

    def forward(self, x):
        if self.quantum_backend == "qiskit" and self.qiskit_map is not None and not self._qiskit_runtime_disabled:
            try:
                qfeat = self.qiskit_map(x)
                return self.qiskit_head(qfeat)
            except Exception as exc:
                if self.qiskit_strict:
                    raise RuntimeError("Qiskit backend runtime failure in strict mode.") from exc
                if not self.qiskit_fallback_on_error:
                    raise RuntimeError("Qiskit backend runtime failure and fallback is disabled.") from exc
                logging.warning(f"Qiskit backend runtime failure ({exc}). Falling back to torch feature map.")
                self._qiskit_runtime_disabled = True

        return self.fallback(x)


class SACCompatibleQuantumBackbone(nn.Module):
    """
    SAC-style MLP with an additive quantum residual.

    This keeps the same high-level feature flow as the SAC MLP while injecting
    active quantum features into the representation consumed by policy/Q heads.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
        quantum_backend: str = "fallback",
        qiskit_num_qubits: int = 6,
        qiskit_reuploads: int = 1,
        qiskit_angle_scale: float = float(np.pi),
        qiskit_seed: int = 1234,
        qiskit_fallback_on_error: bool = True,
        qiskit_strict: bool = False,
        quantum_residual_gain: float = 0.05,
    ):
        super().__init__()
        self.sac_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.quantum = QuantumFeatureBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quantum_backend=quantum_backend,
            qiskit_num_qubits=qiskit_num_qubits,
            qiskit_reuploads=qiskit_reuploads,
            qiskit_angle_scale=qiskit_angle_scale,
            qiskit_seed=qiskit_seed,
            qiskit_fallback_on_error=qiskit_fallback_on_error,
            qiskit_strict=qiskit_strict,
        )
        self.quantum_residual = nn.Linear(hidden_dim, hidden_dim)
        self.quantum_residual_gain = float(quantum_residual_gain)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.quantum_residual.weight)
            self.quantum_residual.bias.zero_()

    def forward(self, x):
        return self.sac_net(x) + self.quantum_residual_gain * self.quantum_residual(self.quantum(x))


class QuantumSquashedGaussianMLPActor(TorchActorModule):
    """
    Drop-in replacement for SquashedGaussianMLPActor with a quantum-ready backbone.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        depth: int = 2,
        quantum_backend: str = "fallback",
        qiskit_num_qubits: int = 6,
        qiskit_reuploads: int = 1,
        qiskit_angle_scale: float = float(np.pi),
        qiskit_seed: int = 1234,
        qiskit_fallback_on_error: bool = True,
        qiskit_strict: bool = False,
        forward_bias_init: float = 1.8,
        brake_bias_init: float = -2.0,
        steer_bias_init: float = 0.0,
        sac_compatible: bool = False,
        quantum_residual_gain: float = 0.05,
    ):
        super().__init__(observation_space, action_space)
        dim_obs, tuple_obs = _obs_dim_and_mode(observation_space)
        self.tuple_obs = tuple_obs

        dim_act = action_space.shape[0]
        self.act_limit = action_space.high[0]

        backbone_cls = SACCompatibleQuantumBackbone if _bool(sac_compatible) else QuantumFeatureBackbone
        self.net = backbone_cls(
            input_dim=dim_obs,
            hidden_dim=hidden_dim,
            depth=depth,
            quantum_backend=quantum_backend,
            qiskit_num_qubits=qiskit_num_qubits,
            qiskit_reuploads=qiskit_reuploads,
            qiskit_angle_scale=qiskit_angle_scale,
            qiskit_seed=qiskit_seed,
            qiskit_fallback_on_error=qiskit_fallback_on_error,
            qiskit_strict=qiskit_strict,
            quantum_residual_gain=quantum_residual_gain,
        )
        self.mu_layer = nn.Linear(hidden_dim, dim_act)
        self.log_std_layer = nn.Linear(hidden_dim, dim_act)
        # Action layout for TrackMania is [gas, brake, steer].
        # Positive gas bias + negative brake bias yields forward starts.
        if dim_act > 0:
            with torch.no_grad():
                self.mu_layer.bias[0] = float(forward_bias_init)
                if dim_act > 1:
                    self.mu_layer.bias[1] = float(brake_bias_init)
                if dim_act > 2:
                    self.mu_layer.bias[2] = float(steer_bias_init)

    def _flatten_obs(self, obs):
        return torch.cat(obs, -1) if self.tuple_obs else torch.flatten(obs, start_dim=1)

    def forward(self, obs, test=False, with_logprob=True):
        x = self._flatten_obs(obs)
        x = self.net(x)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        pi_action = mu if test else pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            action, _ = self.forward(obs, test=test, with_logprob=False)
            res = action.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


class QuantumMLPQFunction(nn.Module):
    def __init__(
        self,
        obs_space,
        act_space,
        hidden_dim: int = 256,
        depth: int = 2,
        quantum_backend: str = "fallback",
        qiskit_num_qubits: int = 6,
        qiskit_reuploads: int = 1,
        qiskit_angle_scale: float = float(np.pi),
        qiskit_seed: int = 1234,
        qiskit_fallback_on_error: bool = True,
        qiskit_strict: bool = False,
        sac_compatible: bool = False,
        quantum_residual_gain: float = 0.05,
    ):
        super().__init__()
        obs_dim, tuple_obs = _obs_dim_and_mode(obs_space)
        self.tuple_obs = tuple_obs
        act_dim = act_space.shape[0]
        backbone_cls = SACCompatibleQuantumBackbone if _bool(sac_compatible) else QuantumFeatureBackbone
        self.net = backbone_cls(
            input_dim=obs_dim + act_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quantum_backend=quantum_backend,
            qiskit_num_qubits=qiskit_num_qubits,
            qiskit_reuploads=qiskit_reuploads,
            qiskit_angle_scale=qiskit_angle_scale,
            qiskit_seed=qiskit_seed,
            qiskit_fallback_on_error=qiskit_fallback_on_error,
            qiskit_strict=qiskit_strict,
            quantum_residual_gain=quantum_residual_gain,
        )
        self.q_out = nn.Linear(hidden_dim, 1)

    def forward(self, obs, act):
        if self.tuple_obs:
            x = torch.cat((*obs, act), -1)
        else:
            x = torch.cat((torch.flatten(obs, start_dim=1), act), -1)
        q = self.q_out(self.net(x))
        return torch.squeeze(q, -1)


class QuantumMLPActorCritic(nn.Module):
    """
    Actor-critic pair compatible with SpinupSacAgent.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        depth: int = 2,
        quantum_backend: str = "fallback",
        qiskit_num_qubits: int = 6,
        qiskit_reuploads: int = 1,
        qiskit_angle_scale: float = float(np.pi),
        qiskit_seed: int = 1234,
        qiskit_fallback_on_error: bool = True,
        qiskit_strict: bool = False,
        forward_bias_init: float = 1.8,
        brake_bias_init: float = -2.0,
        steer_bias_init: float = 0.0,
        sac_compatible: bool = False,
        quantum_residual_gain: float = 0.05,
    ):
        super().__init__()
        self.actor = QuantumSquashedGaussianMLPActor(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dim=hidden_dim,
            depth=depth,
            quantum_backend=quantum_backend,
            qiskit_num_qubits=qiskit_num_qubits,
            qiskit_reuploads=qiskit_reuploads,
            qiskit_angle_scale=qiskit_angle_scale,
            qiskit_seed=qiskit_seed,
            qiskit_fallback_on_error=qiskit_fallback_on_error,
            qiskit_strict=qiskit_strict,
            forward_bias_init=forward_bias_init,
            brake_bias_init=brake_bias_init,
            steer_bias_init=steer_bias_init,
            sac_compatible=sac_compatible,
            quantum_residual_gain=quantum_residual_gain,
        )
        self.q1 = QuantumMLPQFunction(
            obs_space=observation_space,
            act_space=action_space,
            hidden_dim=hidden_dim,
            depth=depth,
            quantum_backend=quantum_backend,
            qiskit_num_qubits=qiskit_num_qubits,
            qiskit_reuploads=qiskit_reuploads,
            qiskit_angle_scale=qiskit_angle_scale,
            qiskit_seed=qiskit_seed,
            qiskit_fallback_on_error=qiskit_fallback_on_error,
            qiskit_strict=qiskit_strict,
            sac_compatible=sac_compatible,
            quantum_residual_gain=quantum_residual_gain,
        )
        self.q2 = QuantumMLPQFunction(
            obs_space=observation_space,
            act_space=action_space,
            hidden_dim=hidden_dim,
            depth=depth,
            quantum_backend=quantum_backend,
            qiskit_num_qubits=qiskit_num_qubits,
            qiskit_reuploads=qiskit_reuploads,
            qiskit_angle_scale=qiskit_angle_scale,
            qiskit_seed=qiskit_seed,
            qiskit_fallback_on_error=qiskit_fallback_on_error,
            qiskit_strict=qiskit_strict,
            sac_compatible=sac_compatible,
            quantum_residual_gain=quantum_residual_gain,
        )

    def act(self, obs, test=False):
        with torch.no_grad():
            action, _ = self.actor(obs, test=test, with_logprob=False)
            res = action.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res
