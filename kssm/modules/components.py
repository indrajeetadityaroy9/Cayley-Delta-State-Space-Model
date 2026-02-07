"""Adaptive components for SC-KSSM (Self-Calibrating KSSM).

This module provides the adaptive components that eliminate manually tuned
hyperparameters while preserving A-stability guarantees:
- AdaptiveTimestep: dt = c / (alpha + |omega|) for scale-invariant dynamics
- SelfNormalizingGates: Layer-aware gate initialization
- Conv1dSiLU: Fused depthwise conv1d + SiLU using CUDA kernel (kernel_size=4)
- compute_variance_preserving_std: T-Fixup style initialization
- apply_spectral_init: Data-driven spectral initialization
- get_adaptive_eps: Dtype-aware epsilon for numerical stability
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from kssm.config import bf16_safety_constants, dynamics_scale
from kssm.ops import conv1d_silu_cuda


# Variance-Preserving Scale (Griffin RG-LRU style)

def compute_variance_preserving_scale(
    alpha: Tensor,
    omega: Tensor,
    dt: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Compute variance-preserving input scale: sqrt(1 - |eigenvalue(A_bar)|^2).

    For the Cayley-discretized A with A = [[-alpha, omega], [-omega, -alpha]]:
        |lambda|^2 = ((1-tau*alpha)^2 + (tau*omega)^2) / ((1+tau*alpha)^2 + (tau*omega)^2)
        scale = sqrt(1 - |lambda|^2) = sqrt(4*tau*alpha / ((1+tau*alpha)^2 + (tau*omega)^2))

    This prevents forward-pass blow-up when eigenvalues approach the unit circle
    (small alpha, large omega), following Griffin's RG-LRU normalization.

    Args:
        alpha: Damping coefficients (batch, seq, n_heads), >= 0
        omega: Frequency coefficients (batch, seq, n_heads)
        dt: Timesteps (batch, seq, n_heads), > 0
        eps: Numerical stability

    Returns:
        scale: (batch, seq, n_heads) in [0, 1]
    """
    tau = dt * 0.5
    tau_alpha = tau * alpha
    tau_omega = tau * omega
    denom = (1.0 + tau_alpha).square() + tau_omega.square()
    scale = torch.sqrt(4.0 * tau_alpha / (denom + eps))
    return scale.clamp(max=1.0)


# Variance-Preserving Initialization (T-Fixup style)

def compute_variance_preserving_std(
    d_model: int,
    d_inner: int,
    n_layers: int,
) -> dict:
    """
    Compute theoretically optimal initialization std for each weight group.

    Based on T-Fixup and Fixup initialization principles:
    - Scale by 1/sqrt(2*n_layers) to prevent gradient explosion/vanishing
    - Use Xavier/He-style fan-in/fan-out scaling
    - Dynamics projection dampened by 1/sqrt(n_layers) (derived from
      variance-preserving analysis: keeps weight perturbations small
      relative to spectral init biases)

    Args:
        d_model: Model dimension
        d_inner: Inner (expanded) dimension
        n_layers: Number of layers

    Returns:
        dict with per-component std values
    """
    # Base std from Xavier
    base_std_in = math.sqrt(2.0 / (d_model + d_inner))
    base_std_inner = math.sqrt(2.0 / (d_inner + d_inner))

    # Layer scaling (T-Fixup style)
    # Each layer contributes to residual, so scale by 1/sqrt(2*n_layers)
    layer_scale = 1.0 / math.sqrt(2 * n_layers)

    # Dynamics projection dampening: derived from depth
    dyn_scale = dynamics_scale(n_layers)

    return {
        "embedding": math.sqrt(1.0 / d_model),
        "in_proj": base_std_in * layer_scale,
        "conv": 1.0 / math.sqrt(4),  # 1/sqrt(kernel_size) for kernel_size=4
        "dynamics_proj": base_std_inner * layer_scale * dyn_scale,
        "Q_proj": 1.0,  # Identity-like, handled separately
        "out_proj": base_std_in * layer_scale,
    }


# AdaptiveTimestep - Natural Frequency Normalization

class AdaptiveTimestep(nn.Module):
    """
    Computes dt adaptively: dt = c / (alpha + |omega| + eps)

    This normalizes dynamics to the "natural frequency" of the system,
    making it inherently scale-invariant. Key theoretical contribution.

    The learned scale factor c is per-head, allowing different heads
    to operate at different effective timescales.

    Mathematical guarantee:
        - dt > 0 always (ratio of positive quantities)
        - dt bounded above by c/eps when alpha=omega=0
        - Adapts to dynamics: faster oscillations -> smaller dt

    Smooth Safety Cap (prevents bf16 precision collapse):
        When omega~0 and tau*alpha~1, computing (1-tau*alpha) in bf16
        suffers catastrophic cancellation. Safety cap constants are
        derived from dtype mantissa precision via bf16_safety_constants().
        For fp32, the cap self-disables (omega_thresh ~ 3e-4).
    """

    def __init__(self, n_heads: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.n_heads = n_heads

        # Derive safety cap constants from dtype precision
        omega_thresh, delta, smoothness = bf16_safety_constants(dtype)
        self.register_buffer("_omega_thresh", torch.tensor(omega_thresh), persistent=False)
        self.register_buffer("_delta", torch.tensor(delta), persistent=False)
        self.register_buffer("_smoothness", torch.tensor(smoothness), persistent=False)

        # Dtype-aware epsilon
        self._eps = get_adaptive_eps(dtype)

        # Learnable per-head scale factor (log-space for positivity)
        # Initialize to log(1) = 0, so softplus gives ~0.69
        self.log_dt_scale = nn.Parameter(torch.zeros(n_heads))

    def forward(self, alpha: Tensor, omega: Tensor) -> Tensor:
        """
        Compute adaptive dt from dynamics.

        Args:
            alpha: (batch, seq, n_heads) - damping coefficients (>= 0)
            omega: (batch, seq, n_heads) - frequency coefficients

        Returns:
            dt: (batch, seq, n_heads) - adaptive timesteps (> 0)
        """
        # Characteristic frequency: sum of damping and oscillation rate
        characteristic_freq = alpha + omega.abs() + self._eps

        # Learned scale (always positive via softplus)
        dt_scale = F.softplus(self.log_dt_scale)

        # dt = scale / frequency (CFL-like condition)
        dt_raw = dt_scale / characteristic_freq

        # Smooth Safety Cap for bf16 precision collapse
        # dt_max = (2 - delta) / alpha ensures tau*alpha < 1 - delta/2
        dt_max = (2.0 - self._delta) / (alpha + self._eps)

        # Smooth blend weight: w -> 1 when |omega| << omega_thresh
        omega_abs = omega.abs()
        w = torch.sigmoid((self._omega_thresh - omega_abs) / self._smoothness)

        # Apply cap smoothly: only engage when omega is small
        dt_capped = torch.minimum(dt_raw, dt_max)
        dt = w * dt_capped + (1.0 - w) * dt_raw

        return dt


# SelfNormalizingGates - Layer-Aware Initialization

class SelfNormalizingGates(nn.Module):
    """
    Gates with layer-aware initialization, no magic biases.

    decay_gate (ZDG): Initialized to protect more at deeper layers
        - Early layers: more decay allowed (learn quickly)
        - Later layers: more protection (preserve information)
        - Formula: protect = 1 - 1/sqrt(layer_idx + 2), self-bounding

    input_gate: Projection with spectral normalization
        - Scale-invariant by construction
        - Learns content-dependent gating from data
    """

    def __init__(
        self,
        d_inner: int,
        n_heads: int,
        layer_idx: int,
        n_layers: int,
    ):
        super().__init__()
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.head_dim = d_inner // n_heads
        self.layer_idx = layer_idx
        self.n_layers = n_layers

        # Decay gate: layer-aware bias + input-dependent projection (Griffin-style)
        self.decay_gate_logit = nn.Parameter(torch.zeros(n_heads))
        self.decay_gate_proj = nn.Linear(d_inner, n_heads, bias=False)

        # Input gate: spectral-normalized projection for scale invariance
        self.input_gate_proj = nn.Linear(d_inner, n_heads, bias=True)
        # Apply spectral normalization for scale invariance
        self.input_gate_proj = torch.nn.utils.spectral_norm(self.input_gate_proj)

        self._init_gates()

    def _init_gates(self):
        """Initialize gates based on layer position.

        protect = 1 - 1/sqrt(layer_idx + 2) is self-bounding:
        layer 0 -> 0.29, layer 11 -> 0.72, layer 100 -> 0.90
        No artificial clamps needed for practical depths.
        """
        with torch.no_grad():
            target_protect = 1.0 - 1.0 / math.sqrt(self.layer_idx + 2)
            # Inverse sigmoid to get logit
            decay_bias = math.log(target_protect / (1 - target_protect))
            self.decay_gate_logit.fill_(decay_bias)

            # Decay gate projection: LeCun fan-in for near-neutral start
            nn.init.normal_(self.decay_gate_proj.weight, std=1.0 / math.sqrt(self.d_inner))

            # Input gate: spectral norm handles scaling, bias starts at 0
            nn.init.zeros_(self.input_gate_proj.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute gates from input features.

        Args:
            x: (batch, seq, d_inner) - input features (typically V_for_dynamics)

        Returns:
            decay_gate: (batch, seq, n_heads) - ZDG protection factor [0, 1]
            input_gate: (batch, seq, n_heads) - input gating factor [0, 1]
        """
        batch, seq_len, _ = x.shape

        # Decay gate: layer-aware base + input-dependent modulation
        decay_gate = torch.sigmoid(self.decay_gate_logit + self.decay_gate_proj(x))

        # Input gate: content-dependent via projection
        input_gate = torch.sigmoid(self.input_gate_proj(x))

        return decay_gate, input_gate


# Conv1dSiLU - Fused Depthwise Conv1d + SiLU using CUDA kernel

class Conv1dSiLU(nn.Module):
    """Depthwise Conv1d + SiLU using CUDA kernel.

    Fixed kernel_size=4 with causal padding. All computation is fused
    into a single CUDA kernel launch with no transposes.
    """

    def __init__(self, d_inner: int):
        super().__init__()
        self.d_inner = d_inner
        self.kernel_size = 4

        # Depthwise conv weights: (d_inner, 1, kernel_size)
        self.weight = nn.Parameter(torch.empty(d_inner, 1, 4))
        self.bias = nn.Parameter(torch.zeros(d_inner))
        self._init_weights()

    def _init_weights(self):
        """Initialize conv weights: Kaiming/He for depthwise with fan_in = kernel_size."""
        with torch.no_grad():
            nn.init.normal_(self.weight, std=1.0 / math.sqrt(self.kernel_size))
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply fused conv1d + SiLU.

        Args:
            x: (batch, seq, d_inner) - input features

        Returns:
            out: (batch, seq, d_inner) - convolved and activated features
        """
        return conv1d_silu_cuda(x, self.weight, self.bias)


# Spectral Initialization for Adaptive Block

def apply_spectral_init(
    block: nn.Module,
    t_min: float,
    t_max: float,
    freq_min: float,
    freq_max: float,
    n_heads: int,
    layer_idx: int,
    n_layers: int,
) -> None:
    """
    Apply log-spaced spectral initialization using calibrated bounds.

    Args:
        block: KSSMBlock to initialize
        t_min, t_max: Calibrated timescale bounds
        freq_min, freq_max: Calibrated frequency bounds
        n_heads: Number of heads
        layer_idx: Layer index (0-indexed)
        n_layers: Total number of layers
    """
    with torch.no_grad():
        # Log-spaced timescales for multi-scale memory
        log_tau = torch.linspace(
            math.log(t_min),
            math.log(t_max),
            n_heads,
        )
        tau = torch.exp(log_tau)

        # Alpha = 1/tau (damping rate inversely proportional to timescale)
        alpha_init = 1.0 / tau

        # Inverse softplus to get bias: softplus(x) = log(1 + exp(x))
        # x = log(exp(alpha) - 1)
        alpha_biases = torch.log(torch.exp(alpha_init) - 1 + 1e-6)

        # Log-spaced frequencies from calibrated bounds
        log_freqs = torch.linspace(
            math.log(freq_min),
            math.log(freq_max),
            n_heads,
        )
        omega_init = torch.exp(log_freqs)

        # Initialize dynamics_proj bias
        # dynamics_proj outputs: [alpha (h), omega (h), omega_mod (h)]
        h = n_heads
        block.dynamics_proj.bias[:h].copy_(alpha_biases)
        # omega starts at calibrated frequencies (no activation)
        block.dynamics_proj.bias[h:2*h].copy_(omega_init)
        # omega_mod starts at zero
        block.dynamics_proj.bias[2*h:3*h].zero_()


# Dtype-Aware Epsilon

def get_adaptive_eps(dtype: torch.dtype) -> float:
    """Compute eps as 100x machine epsilon for the given dtype.

    Args:
        dtype: PyTorch dtype

    Returns:
        Appropriate eps value for numerical stability
    """
    if dtype == torch.float32:
        machine_eps = 1.19e-7
    elif dtype == torch.float16:
        machine_eps = 9.77e-4
    elif dtype == torch.bfloat16:
        machine_eps = 7.81e-3
    else:  # float64
        machine_eps = 2.22e-16

    return 100 * machine_eps


# RMSNorm - Root Mean Square Layer Normalization

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).to(x.dtype) * self.weight
