import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from kssm.config.defaults import KSSMConfig
from .ssd import SSDChunkwiseScan
from .components import (
    AdaptiveTimestep,
    Conv1dSiLU,
    compute_variance_preserving_std,
    compute_variance_preserving_scale_gated,
    apply_spectral_init,
    RMSNorm,
)


class KSSMBlock(nn.Module):
    """Cayley-stable dissipative Hamiltonian SSM block.

    The state matrix A = [[-alpha, omega], [-omega, -alpha]] with alpha >= 0
    is discretized via the Cayley transform, guaranteeing |eigenvalue(A_bar)| <= 1
    unconditionally (A-stability). With alpha > 0 the dynamics are dissipative
    (not exactly symplectic), providing controllable decay alongside rotation.

    Components:
    - AdaptiveTimestep: dt = c/(alpha + |omega|) for scale-invariant dynamics
    - Conv1dSiLU: Fused CUDA depthwise conv1d + SiLU (kernel_size=4)
    - Delta-rule state update with selective erasure via beta gate
    - Learned recurrence gate (Griffin RG-LRU style) for variance-preserving gating
    - Utility gating with state-conditioned feedback for metabolic sparsity
    - Position encoding via omega modulation (RoPE-like)
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

        # Utility Gating (state-conditioned)
        # Predicts "Utility" of current input for the memory
        # Low utility -> Low attention (gate ~ 0) -> "Coasting"
        self.utility_gate = nn.Linear(self.d_inner, self.n_heads, bias=True)
        self.state_gate_proj = nn.Linear(self.n_heads, self.n_heads, bias=False)
        self._metabolic_loss = 0.0  # Store for external access
        self._cached_chunk_states = None  # Cached from previous forward pass

        # Delta-rule update gate (beta)
        self.beta_proj = nn.Linear(self.d_inner, self.n_heads, bias=True)

        # Learned recurrence gate (Griffin RG-LRU style)
        self.recurrence_gate = nn.Linear(self.d_inner, self.n_heads, bias=True)

        # Position encoding frequencies (RoPE-like, not learned)
        freqs = 1.0 / (10000 ** (torch.arange(0, self.n_heads, dtype=torch.float32) / self.n_heads))
        self.register_buffer("rope_freqs", freqs, persistent=False)

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

        # Apply spectral initialization (Universal Priors)
        apply_spectral_init(
            self,
            n_heads=self.n_heads,
            layer_idx=layer_idx,
            n_layers=config.n_layers,
            context_length=config.context_length,
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

        # Beta gate: sigmoid(0) = 0.5 (moderate delta-rule update strength)
        nn.init.normal_(self.beta_proj.weight, std=selection_std)
        nn.init.zeros_(self.beta_proj.bias)

        # Recurrence gate: bias=1.0 so sigmoid(1)~0.73 (moderate decay, mostly open)
        nn.init.normal_(self.recurrence_gate.weight, std=selection_std)
        nn.init.constant_(self.recurrence_gate.bias, 1.0)

        # Utility Gate: Initialize to slightly closed (lazy) state
        # Bias = -1.0 => sigmoid(-1.0) ~= 0.27 (starting with low attention)
        nn.init.normal_(self.utility_gate.weight, std=selection_std)
        nn.init.constant_(self.utility_gate.bias, -1.0)

        # State gate: small init so state feedback starts weak
        nn.init.normal_(self.state_gate_proj.weight, std=0.01)

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

        # Position encoding via omega modulation (RoPE-like)
        positions = torch.arange(seq_len, device=x.device, dtype=omega.dtype).unsqueeze(0).unsqueeze(-1)
        omega = omega + positions * self.rope_freqs

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

        # Dynamics: Alpha is purely predicted by dynamics_proj
        alpha = alpha_base

        # Learned recurrence gate (Griffin RG-LRU style)
        r_gate = torch.sigmoid(self.recurrence_gate(V_for_dynamics_proj))  # (B, L, H)

        # Variance-preserving gating with learned recurrence gate
        # Scale input injection by sqrt(1 - |effective_eigenvalue|^2)
        vp_scale = compute_variance_preserving_scale_gated(alpha, omega, dt, r_gate)
        vp_scale_expanded = vp_scale.unsqueeze(-1).expand(
            -1, -1, -1, self.head_dim
        ).reshape(batch, seq_len, self.d_inner).unsqueeze(-1)
        
        # Base V_gated is just the convolved V
        V_gated = V_conv * vp_scale_expanded

        # State-conditioned utility gating
        # u_gate: (B, L, H) in [0, 1]
        u_logit = self.utility_gate(V_for_dynamics_proj)  # (B, L, H)

        # Add state feedback from previous forward pass (if available)
        if self._cached_chunk_states is not None:
            # Reduce (B, NC, H, D, 2) -> (B, NC, H) via mean energy per head
            state_energy = self._cached_chunk_states.pow(2).sum(-1).mean(-1)
            state_signal = self.state_gate_proj(state_energy)  # (B, NC, H)
            # Upsample from chunk-level to token-level
            state_signal = state_signal.repeat_interleave(
                self.ssd_scan.chunk_size, dim=1
            )[:, :seq_len]  # (B, L, H)
            u_logit = u_logit + state_signal

        u_gate = torch.sigmoid(u_logit)

        # Store mean activation for loss computation (L1 sparsity)
        self._metabolic_loss = u_gate.mean()

        # Modulate V: U = (K * (V * u)) * dt
        # When u ~ 0, input injection is suppressed, preserving existing state (Coasting)
        u_gate_expanded = u_gate.unsqueeze(-1).expand(
            -1, -1, -1, self.head_dim
        ).reshape(batch, seq_len, self.d_inner).unsqueeze(-1)
        
        V_gated = V_gated * u_gate_expanded

        # Q projection
        Q = self.Q_proj(V_for_dynamics_proj).view(batch, seq_len, self.d_inner, 2)

        # Apply input-dependent C selection to Q
        Q = Q * sel_C

        # Delta-rule update gate
        beta = torch.sigmoid(self.beta_proj(V_for_dynamics_proj))  # (B, L, H)

        # Evolution via Cayley discretization (Chunkwise Parallel Scan)
        # Reshape from flat (B,L,d_inner,X) to per-head (B,L,H,D,X) for SSD
        K_heads = K.view(batch, seq_len, self.n_heads, self.head_dim, 2)
        V_gated_heads = V_gated.view(batch, seq_len, self.n_heads, self.head_dim, 1)

        Y, chunk_states = self.ssd_scan(alpha, omega, dt, K_heads, V_gated_heads, beta=beta)

        # Cache chunk states for state-conditioned gating in next forward pass
        self._cached_chunk_states = chunk_states.detach()

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