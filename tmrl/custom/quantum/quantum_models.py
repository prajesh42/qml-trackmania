import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from torch.distributions.normal import Normal

import tmrl.config.config_constants as cfg
from tmrl.actor import TorchActorModule
from tmrl.util import prod


LOG_STD_MAX = 2
LOG_STD_MIN = -20
PARAM_SHIFT = np.pi / 2.0


def _combined_obs_dim(observation_space):
    try:
        obs_dim = sum(prod(s for s in space.shape) for space in observation_space)
        tuple_obs = True
    except TypeError:
        obs_dim = prod(observation_space.shape)
        tuple_obs = False
    return obs_dim, tuple_obs


def _flatten_obs(obs, tuple_obs):
    if tuple_obs:
        return torch.cat(obs, -1)
    return torch.flatten(obs, start_dim=1)


def _mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class _AerCircuitBackend:
    _capability_logged = False

    def __init__(self, n_qubits=4, n_layers=1):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.n_weights = self.n_layers * self.n_qubits * 3

        self.input_params = ParameterVector("x", self.n_qubits)
        self.weight_params = ParameterVector("theta", self.n_weights)

        circuit = QuantumCircuit(self.n_qubits)

        for qubit in range(self.n_qubits):
            circuit.ry(self.input_params[qubit], qubit)

        idx = 0
        for _ in range(self.n_layers):
            for qubit in range(self.n_qubits):
                circuit.rx(self.weight_params[idx], qubit)
                idx += 1
                circuit.ry(self.weight_params[idx], qubit)
                idx += 1
                circuit.rz(self.weight_params[idx], qubit)
                idx += 1
            if self.n_qubits > 1:
                for qubit in range(self.n_qubits - 1):
                    circuit.cx(qubit, qubit + 1)
                circuit.cx(self.n_qubits - 1, 0)

        circuit.save_statevector()
        self.circuit = circuit
        self.available_devices = self._query_available_devices()
        self.requested_device = self._requested_aer_device()
        self.simulator_device = self._select_simulator_device(self.requested_device, self.available_devices)
        self.simulator = AerSimulator(method="statevector", device=self.simulator_device)
        self._z_signs = self._build_z_signs(self.n_qubits)

        if not _AerCircuitBackend._capability_logged:
            logging.info(
                f"Qiskit Aer devices available: {self.available_devices}. "
                f"Requested Aer device: {self.requested_device}. Using: {self.simulator_device}."
            )
            if self.simulator_device != "GPU":
                logging.warning(
                    "QSAC quantum simulator is running on CPU. "
                    "Install GPU-enabled qiskit-aer and set ALG.QUANTUM_AER_DEVICE='GPU' to use GPU Aer."
                )
            _AerCircuitBackend._capability_logged = True

    @staticmethod
    def _requested_aer_device():
        requested = cfg.TMRL_CONFIG.get("ALG", {}).get("QUANTUM_AER_DEVICE", "AUTO")
        requested = str(requested).strip().upper()
        if requested == "CUDA":
            requested = "GPU"
        if requested not in {"AUTO", "CPU", "GPU"}:
            logging.warning(f"Invalid QUANTUM_AER_DEVICE='{requested}'. Falling back to AUTO.")
            requested = "AUTO"
        return requested

    @staticmethod
    def _query_available_devices():
        try:
            devices = AerSimulator(method="statevector").available_devices()
            devices = tuple(str(device).upper() for device in devices)
            if len(devices) == 0:
                return ("CPU",)
            return devices
        except Exception as exc:
            logging.warning(f"Could not query Qiskit Aer devices ({exc}). Assuming CPU only.")
            return ("CPU",)

    @staticmethod
    def _select_simulator_device(requested_device, available_devices):
        if requested_device == "CPU":
            return "CPU"
        if requested_device == "GPU":
            if "GPU" in available_devices:
                return "GPU"
            logging.warning("QUANTUM_AER_DEVICE='GPU' requested but GPU is unavailable in this Aer build.")
            return "CPU"
        # AUTO mode
        return "GPU" if "GPU" in available_devices else "CPU"

    @staticmethod
    def _build_z_signs(n_qubits):
        basis = np.arange(1 << n_qubits, dtype=np.int64)
        z_signs = np.empty((n_qubits, 1 << n_qubits), dtype=np.float64)
        for qubit in range(n_qubits):
            bits = (basis >> qubit) & 1
            z_signs[qubit, :] = 1.0 - 2.0 * bits
        return z_signs

    def _bind_circuit(self, angles, weights):
        bind_map = {}
        for i, param in enumerate(self.input_params):
            bind_map[param] = float(angles[i])
        for i, param in enumerate(self.weight_params):
            bind_map[param] = float(weights[i])
        return self.circuit.assign_parameters(bind_map, inplace=False)

    def evaluate(self, angles, weights):
        bound = self._bind_circuit(angles, weights)
        result = self.simulator.run(bound).result()
        state = np.asarray(result.data(0)["statevector"], dtype=np.complex128)
        probs = np.abs(state) ** 2
        return self._z_signs @ probs

    def evaluate_many(self, angles_batch, weights):
        circuits = [self._bind_circuit(angles, weights) for angles in angles_batch]
        result = self.simulator.run(circuits).result()

        outputs = np.empty((len(circuits), self.n_qubits), dtype=np.float64)
        for idx in range(len(circuits)):
            state = np.asarray(result.data(idx)["statevector"], dtype=np.complex128)
            probs = np.abs(state) ** 2
            outputs[idx, :] = self._z_signs @ probs
        return outputs


class _AerQuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_angles, weights, backend):
        input_cpu = input_angles.detach().cpu()
        weights_cpu = weights.detach().cpu()

        outputs = backend.evaluate_many(input_cpu.numpy(), weights_cpu.numpy())
        out = torch.tensor(outputs, dtype=input_angles.dtype, device=input_angles.device)

        ctx.backend = backend
        ctx.save_for_backward(input_cpu, weights_cpu)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_cpu, weights_cpu = ctx.saved_tensors
        backend = ctx.backend

        need_grad_input, need_grad_weights, _ = ctx.needs_input_grad
        grad_output_cpu = grad_output.detach().cpu().numpy()

        batch_size = input_cpu.shape[0]
        input_np = input_cpu.numpy()
        weights_np = weights_cpu.numpy()

        grad_input_np = None
        if need_grad_input:
            grad_input_np = np.zeros_like(input_np, dtype=np.float64)
            for b in range(batch_size):
                for i in range(backend.n_qubits):
                    plus = input_np[b].copy()
                    minus = input_np[b].copy()
                    plus[i] += PARAM_SHIFT
                    minus[i] -= PARAM_SHIFT
                    eval_plus = backend.evaluate(plus, weights_np)
                    eval_minus = backend.evaluate(minus, weights_np)
                    deriv = 0.5 * (eval_plus - eval_minus)
                    grad_input_np[b, i] = np.dot(grad_output_cpu[b], deriv)

        grad_weights_np = None
        if need_grad_weights:
            grad_weights_np = np.zeros((backend.n_weights,), dtype=np.float64)
            for j in range(backend.n_weights):
                plus = weights_np.copy()
                minus = weights_np.copy()
                plus[j] += PARAM_SHIFT
                minus[j] -= PARAM_SHIFT

                eval_plus = backend.evaluate_many(input_np, plus)
                eval_minus = backend.evaluate_many(input_np, minus)
                deriv = 0.5 * (eval_plus - eval_minus)
                grad_weights_np[j] = np.sum(grad_output_cpu * deriv)

        grad_input = None
        if grad_input_np is not None:
            grad_input = torch.tensor(grad_input_np, dtype=grad_output.dtype, device=grad_output.device)

        grad_weights = None
        if grad_weights_np is not None:
            grad_weights = torch.tensor(grad_weights_np, dtype=grad_output.dtype, device=grad_output.device)

        return grad_input, grad_weights, None


class AerQuantumLayer(nn.Module):
    def __init__(self, n_qubits=4, n_layers=1):
        super().__init__()
        self.backend = _AerCircuitBackend(n_qubits=n_qubits, n_layers=n_layers)
        self.weights = nn.Parameter(0.01 * torch.randn(self.backend.n_weights, dtype=torch.float32))

    def forward(self, input_angles):
        return _AerQuantumFunction.apply(input_angles, self.weights, self.backend)


class FixedAngleEncoder(nn.Module):
    def __init__(self, input_dim, n_qubits, seed=0):
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        scale = 1.0 / math.sqrt(max(1, input_dim))
        projection = torch.randn(input_dim, n_qubits, generator=generator) * scale
        bias = torch.zeros(n_qubits, dtype=torch.float32)
        self.register_buffer("projection", projection)
        self.register_buffer("bias", bias)

    def forward(self, x):
        angles = torch.tanh(x @ self.projection + self.bias) * math.pi
        return angles


class QuantumFeatureExtractor(nn.Module):
    def __init__(self, input_dim, n_qubits=4, n_layers=1, seed=0):
        super().__init__()
        self.encoder = FixedAngleEncoder(input_dim=input_dim, n_qubits=n_qubits, seed=seed)
        self.quantum = AerQuantumLayer(n_qubits=n_qubits, n_layers=n_layers)

    def forward(self, x):
        angles = self.encoder(x)
        return self.quantum(angles)


class SquashedGaussianQuantumActor(TorchActorModule):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__(observation_space, action_space)
        obs_dim, tuple_obs = _combined_obs_dim(observation_space)
        self.tuple_obs = tuple_obs

        alg_cfg = cfg.TMRL_CONFIG["ALG"]
        n_qubits = int(alg_cfg.get("QUANTUM_N_QUBITS", 4))
        n_layers = int(alg_cfg.get("QUANTUM_N_LAYERS", 1))
        seed = int(alg_cfg.get("QUANTUM_ENCODER_SEED", 0))

        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

        self.features = QuantumFeatureExtractor(input_dim=obs_dim, n_qubits=n_qubits, n_layers=n_layers, seed=seed)
        post_sizes = [n_qubits] + list(hidden_sizes)
        self.post_net = _mlp(post_sizes, activation, activation)
        last_dim = hidden_sizes[-1] if len(hidden_sizes) > 0 else n_qubits
        self.mu_layer = nn.Linear(last_dim, dim_act)
        self.log_std_layer = nn.Linear(last_dim, dim_act)
        self.act_limit = act_limit
        self._init_action_priors(dim_act)

    def _init_action_priors(self, dim_act):
        with torch.no_grad():
            if self.mu_layer.bias is not None:
                self.mu_layer.bias.zero_()
                if dim_act >= 1:
                    self.mu_layer.bias[0] = 1.25  # forward by default
                if dim_act >= 2:
                    self.mu_layer.bias[1] = -2.5  # brake unlikely at start
                if dim_act >= 3:
                    self.mu_layer.bias[2] = 0.0
            if self.log_std_layer.bias is not None:
                self.log_std_layer.bias.fill_(-1.5)  # smoother early exploration

    def forward(self, obs, test=False, with_logprob=True):
        x = _flatten_obs(obs, self.tuple_obs)
        x = self.features(x)
        net_out = self.post_net(x)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

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
            a, _ = self.forward(obs, test, False)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


class QuantumQFunction(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim, tuple_obs = _combined_obs_dim(observation_space)
        self.tuple_obs = tuple_obs
        act_dim = action_space.shape[0]
        input_dim = obs_dim + act_dim

        alg_cfg = cfg.TMRL_CONFIG["ALG"]
        n_qubits = int(alg_cfg.get("QUANTUM_N_QUBITS", 4))
        n_layers = int(alg_cfg.get("QUANTUM_N_LAYERS", 1))
        seed = int(alg_cfg.get("QUANTUM_ENCODER_SEED", 0))

        self.features = QuantumFeatureExtractor(input_dim=input_dim, n_qubits=n_qubits, n_layers=n_layers, seed=seed)
        self.q = _mlp([n_qubits] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        obs_flat = _flatten_obs(obs, self.tuple_obs)
        x = torch.cat((obs_flat, act), dim=-1)
        x = self.features(x)
        q = self.q(x)
        return torch.squeeze(q, -1)


class QuantumActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        self.actor = SquashedGaussianQuantumActor(observation_space, action_space, hidden_sizes, activation)
        self.q1 = QuantumQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = QuantumQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res
