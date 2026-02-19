"""Initialization strategies for CDSSM.

- compute_variance_preserving_std: T-Fixup style depth-scaled initialization
- apply_spectral_init: Layer-stratified log-spaced spectral initialization
  for N/2 independent complex eigenvalue pairs per head
"""

import math
import torch
import torch.nn as nn


def compute_variance_preserving_std(
    d_model: int,
    d_inner: int,
    n_layers: int,
) -> dict:
    """Compute theoretically optimal initialization std for each weight group.

    T-Fixup depth scaling: projection weights scale as 1/sqrt(2*n_layers),
    dynamics projections as 1/sqrt(2*n_layers) * 1/sqrt(n_layers). Conv and
    Q_proj use dedicated initialization schemes (Kaiming and identity
    respectively) in their own init methods rather than this dict.
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


def apply_spectral_init(
    block: nn.Module,
    config,
    layer_idx: int,
) -> None:
    """Apply layer-stratified log-spaced spectral initialization.

    Initializes N/2 eigenvalue pairs per head. Each eigenvalue pair j gets
    alpha and omega biases within the layer's spectral band, with slight
    sub-offsets to spread eigenvalues across the band.

    Early layers receive high-frequency (short timescale) priors for local
    pattern matching. Deep layers receive low-frequency (long timescale)
    priors for document-level coherence. Band fraction is derived from
    n_layers: fraction = 2/(n_layers+1) gives exactly 50% overlap between
    adjacent layers.
    """
    n_heads = config.n_heads
    n_layers = config.n_layers
    context_length = config.context_length
    state_dim = config.state_dim
    band_fraction = config.spectral_band_fraction
    eps_log = config.eps_log_argument

    N_half = state_dim // 2
    H = n_heads

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
        # For each eigenvalue pair j, create slightly offset timescale bands
        # This spreads eigenvalues across the band for frequency diversity
        sub_offset = band_width / max(N_half, 1)

        all_alpha_biases = []
        all_omega_biases = []

        for j in range(N_half):
            # Offset within the band for eigenvalue pair j
            j_start = band_start + j * sub_offset / N_half
            j_end = j_start + band_width

            # Log-spaced timescales within this pair's sub-band
            log_tau = torch.linspace(j_start, j_end, H)
            tau = torch.exp(log_tau)

            # Alpha = 1/tau (damping rate inversely proportional to timescale)
            alpha_init = 1.0 / tau

            # Inverse softplus to get bias: softplus(x) = log(1 + exp(x))
            # x = log(exp(alpha) - 1). eps_log prevents log(0).
            alpha_biases = torch.log(torch.exp(alpha_init) - 1 + eps_log)
            all_alpha_biases.append(alpha_biases)

            # Frequencies inversely track timescales within the band
            freq_min_layer = 1.0 / math.exp(j_end)
            freq_max_layer = min(0.5, 1.0 / math.exp(j_start))
            log_freqs = torch.linspace(
                math.log(max(freq_min_layer, eps_log)),
                math.log(freq_max_layer),
                H,
            )
            omega_init = torch.exp(log_freqs)
            all_omega_biases.append(omega_init)

        # Initialize gate_proj bias segments
        # Layout: [alpha_0(H), alpha_1(H), ..., alpha_{N/2-1}(H),
        #          omega_0(H), omega_1(H), ..., omega_{N/2-1}(H), ...]
        for j in range(N_half):
            block.gate_proj.bias[j * H:(j + 1) * H].copy_(all_alpha_biases[j])
            block.gate_proj.bias[(N_half + j) * H:(N_half + j + 1) * H].copy_(all_omega_biases[j])
