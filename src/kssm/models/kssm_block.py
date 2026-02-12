import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from kssm.config.defaults import KSSMConfig
from kssm.ops import cayley_vp_cuda
from .ssd import SSDChunkwiseScan
from .components import (
    AdaptiveTimestep,
    Conv1dSiLU,
    compute_variance_preserving_std,
    apply_spectral_init,
    RMSNorm,
)


class KSSMBlock(nn.Module):
    """Cayley-stable dissipative Hamiltonian SSM block.

    The state matrix A = [[-alpha, omega], [-omega, -alpha]] with alpha >= 0
    is discretized via the Cayley transform, guaranteeing |eigenvalue(A_bar)| <= 1
    unconditionally (A-stability). With alpha > 0 the dynamics are dissipative
    (not exactly symplectic), providing controllable decay alongside rotation.

    Architecture:
    - Decoupled gate pathway: x_gate feeds conv, dynamics, selection, gates
    - Matrix memory: state S ∈ R^(2×D) per head for proper delta-rule retrieval
    - Recurrence gate modulates A_bar eigenvalue (Griffin RG-LRU)
    - L2-normalized keys and queries for delta-rule stability
    - EMA energy feedback for utility gating (no cross-batch leakage)
    - Position encoding via RoPE-like omega modulation
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

        # Derived gating range: c = log(context_length)
        self._gating_c = math.log(config.context_length)

        # in_proj: [z_gate, x_gate, K, V]
        # z: d_inner (output gate)
        # x_gate: d_inner (feeds conv -> dynamics/selection/gates)
        # K: d_inner (D-dimensional keys per head)
        # V: n_heads * 2 (2D Hamiltonian value vectors per head)
        in_proj_size = self.d_inner * 3 + self.n_heads * 2
        self.in_proj = nn.Linear(self.d_model, in_proj_size, bias=True)

        # Fused conv1d + SiLU (CUDA kernel) — on gate pathway
        self.conv = Conv1dSiLU(self.d_inner)

        # Dynamics projection: alpha + omega (2 outputs per head)
        self.dynamics_proj = nn.Linear(
            self.d_inner,
            self.n_heads * 2,
            bias=True,
        )

        # Adaptive timestep
        self.adaptive_dt = AdaptiveTimestep(self.n_heads)

        # SSD Scanner (Mamba-2 algorithm)
        self.ssd_scan = SSDChunkwiseScan(self.d_inner, self.n_heads)

        # Utility Gating (state-conditioned via EMA energy)
        self.utility_gate = nn.Linear(self.d_inner, self.n_heads, bias=True)
        self.state_gate_proj = nn.Linear(self.n_heads, self.n_heads, bias=False)
        self._metabolic_loss = 0.0
        self.register_buffer('_ema_energy', torch.zeros(self.n_heads), persistent=False)

        # Delta-rule update gate (beta)
        self.beta_proj = nn.Linear(self.d_inner, self.n_heads, bias=True)

        # Learned recurrence gate (Griffin RG-LRU style)
        self.recurrence_gate = nn.Linear(self.d_inner, self.n_heads, bias=True)

        # Position encoding frequencies (RoPE-like, not learned)
        freqs = 1.0 / (10000 ** (torch.arange(0, self.n_heads, dtype=torch.float32) / self.n_heads))
        self.register_buffer("rope_freqs", freqs, persistent=False)

        # Input-dependent selection (Mamba-style) — per-head scalars
        self.selection_B = nn.Linear(self.d_inner, self.n_heads, bias=False)
        self.selection_C = nn.Linear(self.d_inner, self.n_heads, bias=False)
        self.selection_dt = nn.Linear(self.d_inner, self.n_heads, bias=False)

        # Q projection: D-dimensional query per head
        self.Q_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)

        # Readout projection: maps retrieved 2-vectors back to feature space
        self.readout_proj = nn.Linear(self.n_heads * 2, self.d_inner, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        self.norm = RMSNorm(self.d_model)
        self.ssm_norm = nn.GroupNorm(config.ssm_norm_groups, self.d_inner)
        self.D = nn.Parameter(torch.zeros(self.d_inner))

        # Initialize weights
        self._init_weights()

        # Apply spectral initialization (returns per-head timescales)
        tau_h = apply_spectral_init(
            self,
            n_heads=self.n_heads,
            layer_idx=layer_idx,
            n_layers=config.n_layers,
            context_length=config.context_length,
        )

        # Per-head EMA decay aligned to spectral timescale
        # Fast heads (τ≈1) → decay≈0.9, slow heads (τ≈8192) → decay≈0.999
        ema_decay = (1.0 - 1.0 / tau_h).clamp(0.9, 0.999)
        self.register_buffer('_ema_decay', ema_decay, persistent=False)

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
            # x_gate: standard init (already done by normal_ above)
            # K: standard init, D-dimensional
            k_start = self.d_inner * 2
            k_end = k_start + self.d_inner
            self.in_proj.weight[k_start:k_end, :].normal_(std=stds["in_proj"])
            # V: n_heads * 2, small init
            v_start = k_end
            v_end = v_start + self.n_heads * 2
            self.in_proj.weight[v_start:v_end, :].normal_(
                std=stds["in_proj"] / math.sqrt(self.head_dim)
            )

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
        nn.init.normal_(self.utility_gate.weight, std=selection_std)
        nn.init.constant_(self.utility_gate.bias, -1.0)

        # State gate: small init so state feedback starts weak
        nn.init.normal_(self.state_gate_proj.weight, std=0.01)

        # Q_proj: identity-like initialization
        nn.init.eye_(self.Q_proj.weight)

        # Readout projection
        nn.init.normal_(self.readout_proj.weight, std=1.0 / math.sqrt(self.n_heads * 2))

        nn.init.normal_(self.out_proj.weight, std=stds["out_proj"])
        nn.init.zeros_(self.D)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        batch, seq_len, _ = x.shape
        residual = x

        x = self.norm(x)
        proj = self.in_proj(x)

        # Split into z_gate, x_gate, K, V
        z, x_gate, K, V = proj.split(
            [self.d_inner, self.d_inner, self.d_inner, self.n_heads * 2],
            dim=-1,
        )
        K = K.view(batch, seq_len, self.n_heads, self.head_dim)  # (B, L, H, D)
        V = V.view(batch, seq_len, self.n_heads, 2)               # (B, L, H, 2)

        # Conv on gate pathway (decoupled from V)
        x_conv = self.conv(x_gate)  # (B, L, d_inner)

        # Compute dynamics from gate pathway
        x_conv_cast = x_conv.to(self.dynamics_proj.weight.dtype)
        dynamics = self.dynamics_proj(x_conv_cast)

        # Parse dynamics: alpha, omega
        h = self.n_heads

        # Alpha (softplus for positivity)
        alpha_base = F.softplus(dynamics[..., :h])

        # Omega
        omega = dynamics[..., h:2*h]

        # Position encoding via omega modulation (RoPE-like)
        positions = torch.arange(seq_len, device=x.device, dtype=omega.dtype).unsqueeze(0).unsqueeze(-1)
        omega = omega + positions * self.rope_freqs

        # Adaptive timestep
        dt = self.adaptive_dt(alpha_base, omega)

        # Input-dependent selection (Mamba-style): per-head scalars
        sel_B = self.selection_B(x_conv_cast).unsqueeze(-1)  # (B, L, H, 1)
        K = K * sel_B  # (B, L, H, D) * (B, L, H, 1)

        sel_C = self.selection_C(x_conv_cast).unsqueeze(-1)  # (B, L, H, 1)

        sel_dt = F.softplus(self.selection_dt(x_conv_cast))
        dt = dt + sel_dt

        # Dynamics: Alpha is purely predicted by dynamics_proj
        alpha = alpha_base

        # Learned recurrence gate (Griffin RG-LRU style)
        r_gate = torch.sigmoid(self.recurrence_gate(x_conv_cast))  # (B, L, H)

        # State-conditioned utility gating via EMA energy
        state_signal = self.state_gate_proj(self._ema_energy.clone().detach())  # (H,)
        u_logit = self.utility_gate(x_conv_cast) + state_signal  # broadcast to (B, L, H)

        u_gate = torch.sigmoid(u_logit)

        # Store mean activation for loss computation (L1 sparsity)
        self._metabolic_loss = u_gate.mean()

        # L2-normalize keys for delta-rule stability
        K = F.normalize(K, dim=-1)

        # Q projection: D-dimensional query per head
        Q = self.Q_proj(x_conv_cast).view(batch, seq_len, self.n_heads, self.head_dim)
        Q = Q * sel_C  # (B, L, H, D) * (B, L, H, 1)

        # L2-normalize queries for delta-rule stability
        Q = F.normalize(Q, dim=-1)

        # Delta-rule update gate
        beta = torch.sigmoid(self.beta_proj(x_conv_cast))  # (B, L, H)

        # Fused Cayley discretization + recurrence gate + VP scale (CUDA kernel)
        A_bar, vp_scale = cayley_vp_cuda(alpha, omega, dt, r_gate, self._gating_c)

        # Apply VP scale and utility gate to V before scan
        V_gated = V * (vp_scale * u_gate).unsqueeze(-1)  # (B, L, H, 2)

        # Chunkwise Parallel Scan (CUDA kernels for intra/inter-chunk)
        Y, chunk_states = self.ssd_scan(A_bar, K, V_gated, beta)

        # Update EMA energy from chunk states
        if self.training:
            # chunk_states: (B, NC, H, 2, D) -> energy per head
            current_energy = chunk_states.pow(2).sum(dim=(-1, -2)).mean(dim=(0, 1))  # (H,)
            self._ema_energy.copy_(self._ema_decay * self._ema_energy + (1.0 - self._ema_decay) * current_energy.detach())

        # Readout: retrieve 2-vectors from matrix state using Q
        # Y: (B, L, H, 2, D), Q: (B, L, H, D) -> retrieved: (B, L, H, 2)
        retrieved = torch.einsum('blhsd,blhd->blhs', Y, Q)

        # Project retrieved 2-vectors back to feature space
        y = self.readout_proj(retrieved.reshape(batch, seq_len, -1))  # (B, L, d_inner)

        # GroupNorm in fp32 for numerical stability
        y_dtype = y.dtype
        y_for_norm = y.float().transpose(1, 2)
        y = F.group_norm(
            y_for_norm, self.ssm_norm.num_groups,
            self.ssm_norm.weight.float(), self.ssm_norm.bias.float(),
            self.ssm_norm.eps,
        ).to(y_dtype).transpose(1, 2)
        y = y * F.silu(z)
        y = y + self.D * x_conv

        y = y.to(self.out_proj.weight.dtype)
        output = self.out_proj(y)
        output = residual + output

        return output
