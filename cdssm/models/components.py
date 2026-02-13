"""Adaptive components for CDSSM.

This module provides the adaptive components with all constants derived
from the CDSSMConfig (no hardcoded magic numbers):
- Conv1dSiLU: Fused depthwise conv1d + SiLU using CUDA kernel
- compute_variance_preserving_std: T-Fixup style initialization
- apply_spectral_init: Layer-stratified spectral initialization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from cdssm.ops import conv1d_silu_cuda


# Variance-Preserving Initialization (T-Fixup style)

def compute_variance_preserving_std(
    d_model: int,
    d_inner: int,
    n_layers: int,
) -> dict:
    """Compute theoretically optimal initialization std for each weight group.

    T-Fixup depth scaling: projection weights scale as 1/sqrt(2*n_layers),
    dynamics projections as 1/sqrt(n_layers). Conv and Q_proj use dedicated
    initialization schemes (Kaiming and identity respectively) in their own
    init methods rather than this dict.
    """
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
        "dynamics_proj": base_std_inner * layer_scale * dyn_scale,
        "out_proj": base_std_in * layer_scale,
    }



# Conv1dSiLU - Fused Depthwise Conv1d + SiLU using CUDA kernel

class Conv1dSiLU(nn.Module):
    """Depthwise Conv1d + SiLU using CUDA kernel."""

    def __init__(self, d_inner: int, kernel_size: int = 4):
        super().__init__()
        self.d_inner = d_inner
        self.kernel_size = kernel_size

        # Depthwise conv weights: (d_inner, 1, kernel_size)
        self.weight = nn.Parameter(torch.empty(d_inner, 1, kernel_size))
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
    config,
    layer_idx: int,
) -> Tensor:
    """Apply layer-stratified log-spaced spectral initialization.

    Early layers receive high-frequency (short timescale) priors for local
    pattern matching. Deep layers receive low-frequency (long timescale)
    priors for document-level coherence. Band fraction is derived from
    n_layers: fraction = 2/(n_layers+1) gives exactly 50% overlap between
    adjacent layers.

    Args:
        block: CDSSMBlock module with gate_proj attribute.
        config: CDSSMConfig with n_heads, n_layers, context_length,
            spectral_band_fraction, eps_log_argument.
        layer_idx: Layer index (0-based).

    Returns:
        tau: (n_heads,) per-head timescales for EMA decay derivation.
    """
    n_heads = config.n_heads
    n_layers = config.n_layers
    context_length = config.context_length
    band_fraction = config.spectral_band_fraction
    eps_log = config.eps_log_argument

    # Universal Bounds
    t_min = 1.0
    t_max = float(context_length)

    log_t_min = math.log(t_min)
    log_t_max = math.log(t_max)
    log_range = log_t_max - log_t_min

    # Layer-dependent band: slides from short to long timescales with depth
    layer_frac = layer_idx / max(n_layers - 1, 1)  # 0.0 to 1.0
    band_width = band_fraction * log_range
    band_start = log_t_min + layer_frac * (log_range - band_width)
    band_end = band_start + band_width

    with torch.no_grad():
        # Log-spaced timescales within this layer's band
        log_tau = torch.linspace(band_start, band_end, n_heads)
        tau = torch.exp(log_tau)

        # Alpha = 1/tau (damping rate inversely proportional to timescale)
        alpha_init = 1.0 / tau

        # Inverse softplus to get bias: softplus(x) = log(1 + exp(x))
        # x = log(exp(alpha) - 1). eps_log prevents log(0).
        alpha_biases = torch.log(torch.exp(alpha_init) - 1 + eps_log)

        # Frequencies inversely track timescales within the band
        freq_min_layer = 1.0 / math.exp(band_end)
        freq_max_layer = min(0.5, 1.0 / math.exp(band_start))  # Nyquist = 0.5 (Shannon)
        log_freqs = torch.linspace(
            math.log(max(freq_min_layer, eps_log)),
            math.log(freq_max_layer),
            n_heads,
        )
        omega_init = torch.exp(log_freqs)

        # Initialize gate_proj bias (alpha and omega segments)
        h = n_heads
        block.gate_proj.bias[:h].copy_(alpha_biases)
        block.gate_proj.bias[h:2*h].copy_(omega_init)

    return tau


# RMSNorm - Root Mean Square Layer Normalization

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        eps: Normalization epsilon. Default derived from io_eps² (BF16 gradient
            representability constraint: rsqrt gradient O(eps^{-1/2}) < 1/io_eps).
    """

    def __init__(self, d_model: int, eps: float = 6.103515625e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).to(x.dtype) * self.weight
