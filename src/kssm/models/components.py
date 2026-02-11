"""Adaptive components for SC-KSSM (Self-Calibrating KSSM).

This module provides the adaptive components that eliminate manually tuned
hyperparameters while preserving A-stability guarantees:
- AdaptiveTimestep: dt = c / (alpha + |omega|) for scale-invariant dynamics
- Conv1dSiLU: Fused depthwise conv1d + SiLU using CUDA kernel (kernel_size=4)
- compute_variance_preserving_std: T-Fixup style initialization
- apply_spectral_init: Data-driven spectral initialization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from kssm.ops import conv1d_silu_cuda


def bf16_safety_constants(dtype: torch.dtype) -> tuple[float, float, float]:
    """Derive smooth safety cap constants from dtype mantissa precision."""
    eps = torch.finfo(dtype).eps
    omega_thresh = math.sqrt(eps)
    delta = 16.0 * eps
    smoothness = omega_thresh / 5.0
    return omega_thresh, delta, smoothness



# Variance-Preserving Scale (Griffin RG-LRU style)

def compute_variance_preserving_scale_gated(
    alpha: Tensor,
    omega: Tensor,
    dt: Tensor,
    r_gate: Tensor,
    c: float = 8.0,
    eps: float = 1e-6,
) -> Tensor:
    """Compute learnable VP scale with Griffin RG-LRU-style recurrence gate.

    The learned gate r_t modulates the effective eigenvalue decay via
    exponentiation: effective_eig = eig^(c * r_t). When r_t -> 0, the
    effective decay is minimal (state held). When r_t -> 1, the decay
    is amplified (state flushed). This gives the model control over
    retention independent of the Hamiltonian dynamics.

    Args:
        alpha, omega, dt: (B, L, H) dynamics parameters
        r_gate: (B, L, H) learned recurrence gate in [0, 1]
        c: gating range constant (default 8.0, matching Griffin)
        eps: numerical stability constant
    """
    tau = dt * 0.5
    tau_alpha = tau * alpha
    tau_omega = tau * omega

    # |eigenvalue(A_bar)|^2 = ((1-tau*alpha)^2 + (tau*omega)^2) / ((1+tau*alpha)^2 + (tau*omega)^2)
    numer = (1.0 - tau_alpha).square() + tau_omega.square()
    denom = (1.0 + tau_alpha).square() + tau_omega.square()
    eig_sq = numer / (denom + eps)

    # Modulate decay via learned gate
    effective_eig_sq = eig_sq.pow(c * r_gate)

    scale = torch.sqrt((1.0 - effective_eig_sq).clamp(min=eps))
    return scale.clamp(max=1.0)


# Variance-Preserving Initialization (T-Fixup style)

def compute_variance_preserving_std(
    d_model: int,
    d_inner: int,
    n_layers: int,
) -> dict:
    """Compute theoretically optimal initialization std for each weight group."""
    # Base std from Xavier
    base_std_in = math.sqrt(2.0 / (d_model + d_inner))
    base_std_inner = math.sqrt(2.0 / (d_inner + d_inner))

    # Layer scaling (T-Fixup style)
    # Each layer contributes to residual, so scale by 1/sqrt(2*n_layers)
    layer_scale = 1.0 / math.sqrt(2 * n_layers)

    # Dynamics projection dampening: derived from depth
    dyn_scale = 1.0 / math.sqrt(n_layers)

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
    """Computes dt adaptively: dt = c / (alpha + |omega| + eps)."""

    def __init__(self, n_heads: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.n_heads = n_heads

        # Derive safety cap constants from dtype precision
        omega_thresh, delta, smoothness = bf16_safety_constants(dtype)
        self.register_buffer("_omega_thresh", torch.tensor(omega_thresh), persistent=False)
        self.register_buffer("_delta", torch.tensor(delta), persistent=False)
        self.register_buffer("_smoothness", torch.tensor(smoothness), persistent=False)

        # Dtype-aware epsilon
        self._eps = torch.finfo(dtype).eps * 100

        # Learnable per-head scale factor (log-space for positivity)
        # Initialize to log(1) = 0, so softplus gives ~0.69
        self.log_dt_scale = nn.Parameter(torch.zeros(n_heads))

    def forward(self, alpha: Tensor, omega: Tensor) -> Tensor:
        """Compute adaptive dt from dynamics."""
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


# Conv1dSiLU - Fused Depthwise Conv1d + SiLU using CUDA kernel

class Conv1dSiLU(nn.Module):
    """Depthwise Conv1d + SiLU using CUDA kernel."""

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
        return conv1d_silu_cuda(x, self.weight, self.bias)


# Spectral Initialization for Adaptive Block (Universal Prior)

def apply_spectral_init(
    block: nn.Module,
    n_heads: int,
    layer_idx: int,
    n_layers: int,
    context_length: int = 8192,
) -> Tensor:
    """Apply layer-stratified log-spaced spectral initialization.

    Early layers receive high-frequency (short timescale) priors for local
    pattern matching. Deep layers receive low-frequency (long timescale)
    priors for document-level coherence. Each layer covers an overlapping
    50% band of the total log-timescale range, sliding with depth.

    Returns:
        tau: (n_heads,) per-head timescales for EMA decay derivation.
    """
    # Universal Bounds
    t_min = 1.0
    t_max = float(context_length)

    log_t_min = math.log(t_min)
    log_t_max = math.log(t_max)
    log_range = log_t_max - log_t_min

    # Layer-dependent band: slides from short to long timescales with depth
    layer_frac = layer_idx / max(n_layers - 1, 1)  # 0.0 to 1.0
    band_width = 0.5 * log_range  # each layer sees 50% of full range
    band_start = log_t_min + layer_frac * (log_range - band_width)
    band_end = band_start + band_width

    with torch.no_grad():
        # Log-spaced timescales within this layer's band
        log_tau = torch.linspace(band_start, band_end, n_heads)
        tau = torch.exp(log_tau)

        # Alpha = 1/tau (damping rate inversely proportional to timescale)
        alpha_init = 1.0 / tau

        # Inverse softplus to get bias: softplus(x) = log(1 + exp(x))
        # x = log(exp(alpha) - 1)
        alpha_biases = torch.log(torch.exp(alpha_init) - 1 + 1e-6)

        # Frequencies inversely track timescales within the band
        freq_min_layer = 1.0 / math.exp(band_end)
        freq_max_layer = min(0.5, 1.0 / math.exp(band_start))
        log_freqs = torch.linspace(
            math.log(max(freq_min_layer, 1e-8)),
            math.log(freq_max_layer),
            n_heads,
        )
        omega_init = torch.exp(log_freqs)

        # Initialize dynamics_proj bias
        h = n_heads
        block.dynamics_proj.bias[:h].copy_(alpha_biases)
        block.dynamics_proj.bias[h:2*h].copy_(omega_init)

    return tau


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
