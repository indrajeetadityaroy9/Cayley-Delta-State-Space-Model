"""KSSM Block: Kinetic State Space Model with Cayley discretization.

Core Evolution Equation:
    H_t = A_bar @ H_{t-1} + (K * V) * dt
    y_t = Q^T @ H_t

Cayley Transform (A-stable discretization):
    A_bar = (I - tau*A)^{-1}(I + tau*A), where tau = dt/2

    For A = [[-alpha, omega], [-omega, -alpha]] with alpha >= 0,
    the Cayley transform guarantees |eigenvalue(A_bar)| <= 1.

Key Components:
    - AdaptiveTimestep: dt = c/(alpha + |omega|) for scale-invariance
    - SelfNormalizingGates: Layer-aware protect/input gates
    - Conv1dSiLU: Fused CUDA kernel (kernel_size=4)

See: kssm/csrc/include/cayley_math.cuh for implementation details.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from kssm.config import KSSMConfig
from kssm.modules.ssd import SSDChunkwiseScan
from kssm.modules.components import (
    AdaptiveTimestep,
    SelfNormalizingGates,
    Conv1dSiLU,
    compute_variance_preserving_std,
    compute_variance_preserving_scale,
    apply_spectral_init,
    RMSNorm,
)


class KSSMBlock(nn.Module):
    """SC-KSSM block with adaptive timestep, convolution, and gates.

    Uses:
    - AdaptiveTimestep: dt = c/(alpha + |omega|)
    - Conv1dSiLU: Fused CUDA kernel (kernel_size=4)
    - SelfNormalizingGates: Variance-preserving gates
    - Input-dependent selection (Mamba-style)
    - Variance-preserving gating (Griffin RG-LRU)
    - Phase Modulation Encoding: omega_mod for PME
    """

    def __init__(self, config: KSSMConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_layers = config.n_layers

        # Dynamics parameterization: alpha, omega, omega_mod (PME)
        self._n_dynamics = 3

        # in_proj: [z_gate, K, V]
        in_proj_size = self.d_inner + self.d_inner * 2 + self.d_inner
        self.in_proj = nn.Linear(self.d_model, in_proj_size, bias=True)

        # Fused conv1d + SiLU (CUDA kernel)
        self.conv = Conv1dSiLU(self.d_inner)

        # Dynamics projection
        self.dynamics_proj = nn.Linear(
            self.d_inner,
            self.n_heads * self._n_dynamics,
            bias=True,
        )

        # Adaptive timestep
        self.adaptive_dt = AdaptiveTimestep(self.n_heads)

        # SSD Scanner (Mamba-2 algorithm)
        self.ssd_scan = SSDChunkwiseScan(self.d_inner, self.n_heads)

        # ZDG: Zero-Damping Gate with layer-aware initialization
        self.gates = SelfNormalizingGates(
            self.d_inner,
            self.n_heads,
            layer_idx,
            config.n_layers,
        )

        # Input-dependent selection (Mamba-style)
        self.selection_B = nn.Linear(self.d_inner, self.n_heads * 2, bias=False)
        self.selection_C = nn.Linear(self.d_inner, self.n_heads * 2, bias=False)
        self.selection_dt = nn.Linear(self.d_inner, self.n_heads, bias=False)

        # Q projection and output
        self.Q_proj = nn.Linear(self.d_inner, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        self.norm = RMSNorm(self.d_model)
        self.ssm_norm = nn.GroupNorm(config.ssm_norm_groups, self.d_inner)
        self.D = nn.Parameter(torch.zeros(self.d_inner))

        # Initialize weights
        self._init_weights()

        # Apply spectral initialization (uses derived defaults if not calibrated)
        apply_spectral_init(
            self,
            t_min=config.t_min,
            t_max=config.t_max,
            freq_min=config.freq_min,
            freq_max=config.freq_max,
            n_heads=self.n_heads,
            layer_idx=layer_idx,
            n_layers=config.n_layers,
        )

    def _init_weights(self):
        """Variance-preserving initialization."""
        stds = compute_variance_preserving_std(
            self.d_model,
            self.d_inner,
            self.n_layers,
        )

        nn.init.normal_(self.in_proj.weight, std=stds["in_proj"])
        nn.init.zeros_(self.in_proj.bias)

        with torch.no_grad():
            self.in_proj.bias[:self.d_inner].fill_(0.0)  # Z gate: 50% open
            # K: fan-out correction for 2 state dimensions
            k_start = self.d_inner
            k_end = k_start + self.d_inner * 2
            self.in_proj.weight[k_start:k_end, :].normal_(std=stds["in_proj"] * math.sqrt(2))
            # V: single state dimension, no correction
            v_start = k_end
            v_end = v_start + self.d_inner
            self.in_proj.weight[v_start:v_end, :].normal_(std=stds["in_proj"])

        nn.init.normal_(self.dynamics_proj.weight, std=stds["dynamics_proj"])
        nn.init.zeros_(self.dynamics_proj.bias)

        # Selection projections: LeCun fan-in scaling for near-neutral start
        selection_std = 1.0 / math.sqrt(self.d_inner)
        nn.init.normal_(self.selection_B.weight, std=selection_std)
        nn.init.normal_(self.selection_C.weight, std=selection_std)
        nn.init.normal_(self.selection_dt.weight, std=selection_std)

        # Q_proj: identity-like readout (both h0 and h1 dimensions)
        nn.init.zeros_(self.Q_proj.weight)
        with torch.no_grad():
            W = self.Q_proj.weight.view(self.d_inner, 2, self.d_inner)
            for i in range(self.d_inner):
                W[i, 0, i] = 1.0
                W[i, 1, i] = 1.0

        nn.init.normal_(self.out_proj.weight, std=stds["out_proj"])
        nn.init.zeros_(self.D)  # Zero so evolution path receives gradients

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        batch, seq_len, _ = x.shape
        residual = x

        x = self.norm(x)
        proj = self.in_proj(x)

        # Split into z_gate, K, V
        z, K, V = proj.split(
            [self.d_inner, self.d_inner * 2, self.d_inner],
            dim=-1,
        )
        K = K.view(batch, seq_len, self.d_inner, 2)
        V = V.view(batch, seq_len, self.d_inner, 1)

        # Adaptive convolution (includes SiLU)
        V_conv = self.conv(V.view(batch, seq_len, -1))
        V_conv = V_conv.view(batch, seq_len, self.d_inner, 1)

        # Compute dynamics from V
        V_for_dynamics = V_conv.squeeze(-1)
        V_for_dynamics_proj = V_for_dynamics.to(self.dynamics_proj.weight.dtype)
        dynamics = self.dynamics_proj(V_for_dynamics_proj)

        # Parse dynamics: alpha, omega, omega_mod
        h = self.n_heads

        # Alpha (softplus for positivity)
        alpha_base = F.softplus(dynamics[..., :h])

        # Omega
        omega_base = dynamics[..., h:2*h]

        # Phase Modulation Encoding
        omega_mod = dynamics[..., 2*h:3*h]
        omega = omega_base + omega_mod

        # Adaptive timestep
        dt = self.adaptive_dt(alpha_base, omega)

        # Input-dependent selection (Mamba-style): modulate K, Q, dt from input
        sel_B = self.selection_B(V_for_dynamics_proj)
        sel_B = sel_B.view(batch, seq_len, self.n_heads, 1, 2)
        sel_B = sel_B.expand(-1, -1, -1, self.head_dim, -1).reshape(
            batch, seq_len, self.d_inner, 2
        )
        K = K * sel_B  # Input-dependent B modulation

        sel_C = self.selection_C(V_for_dynamics_proj)
        sel_C = sel_C.view(batch, seq_len, self.n_heads, 1, 2)
        sel_C = sel_C.expand(-1, -1, -1, self.head_dim, -1).reshape(
            batch, seq_len, self.d_inner, 2
        )

        sel_dt = F.softplus(self.selection_dt(V_for_dynamics_proj))
        dt = dt + sel_dt  # Input-dependent Delta modulation

        # ZDG: Zero-Damping Gate
        protect_gate, input_gate = self.gates(V_for_dynamics_proj)
        alpha = alpha_base * (1.0 - protect_gate)

        # Apply input gate to V
        input_gate_expanded = input_gate.unsqueeze(-1).expand(
            -1, -1, -1, self.head_dim
        ).reshape(batch, seq_len, self.d_inner).unsqueeze(-1)
        V_gated = V_conv * input_gate_expanded

        # Variance-preserving gating (Griffin RG-LRU style)
        # Scale input injection by sqrt(1 - |eigenvalue(A_bar)|^2)
        vp_scale = compute_variance_preserving_scale(alpha, omega, dt)
        vp_scale_expanded = vp_scale.unsqueeze(-1).expand(
            -1, -1, -1, self.head_dim
        ).reshape(batch, seq_len, self.d_inner).unsqueeze(-1)
        V_gated = V_gated * vp_scale_expanded

        # Q projection
        Q = self.Q_proj(V_for_dynamics_proj).view(batch, seq_len, self.d_inner, 2)

        # Apply input-dependent C selection to Q
        Q = Q * sel_C

        # Evolution via Cayley discretization (Chunkwise Parallel Scan)
        # Reshape from flat (B,L,d_inner,X) to per-head (B,L,H,D,X) for SSD
        K_heads = K.view(batch, seq_len, self.n_heads, self.head_dim, 2)
        V_gated_heads = V_gated.view(batch, seq_len, self.n_heads, self.head_dim, 1)

        Y = self.ssd_scan(alpha, omega, dt, K_heads, V_gated_heads)

        # Reshape SSD output (B, L, H, D, 2) back to (B, L, d_inner, 2) for Q readout
        Y = Y.reshape(batch, seq_len, self.d_inner, 2)
        
        # y = Q^T @ H
        # Q: (B, L, d_inner, 2)
        # Y (state): (B, L, d_inner, 2)
        # Dot product over the last dimension (2)
        y = (Q * Y).sum(dim=-1) # (B, L, d_inner)

        # GroupNorm in fp32 for numerical stability (matches RMSNorm upcast pattern)
        y_for_norm = y.float().transpose(1, 2)
        y = self.ssm_norm(y_for_norm).to(y.dtype).transpose(1, 2)
        y = y * F.silu(z)
        y = y + self.D * V_conv.view(batch, seq_len, -1)

        y = y.to(self.out_proj.weight.dtype)
        output = self.out_proj(y)
        output = residual + output

        return output
