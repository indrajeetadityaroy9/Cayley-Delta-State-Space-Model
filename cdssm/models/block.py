import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cdssm.config.defaults import CDSSMConfig
from cdssm.ops import dynamics_fused_cuda, normalize_kq_cuda
from cdssm.models.ssd import SSDChunkwiseScan
from cdssm.models.components import (
    Conv1dSiLU,
    compute_variance_preserving_std,
    apply_spectral_init,
    RMSNorm,
)


class CDSSMBlock(nn.Module):
    """Cayley-stable dissipative Hamiltonian SSM block.

    The state matrix A = [[-alpha, omega], [-omega, -alpha]] with alpha >= 0
    is discretized via the Cayley transform, guaranteeing |eigenvalue(A_bar)| <= 1
    unconditionally (A-stability). With alpha > 0 the dynamics are dissipative
    (not exactly symplectic), providing controllable decay alongside rotation.

    Architecture:
    - Fused gate projection: all per-head scalars in one GEMM
    - Matrix memory: state S in R^(2xD) per head for delta-rule retrieval
    - Recurrence gate modulates A_bar eigenvalue (Griffin RG-LRU)
    - L2-normalized keys and queries for delta-rule stability
    - sel_B/sel_C as write/read strength gates (post-normalization)
    - Position encoding via RoPE-like omega modulation

    All constants derived from CDSSMConfig — no hardcoded magic numbers.
    """

    def __init__(self, config: CDSSMConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config

        # in_proj: [z_gate, x_gate, K, V]
        in_proj_size = config.d_inner * 3 + config.n_heads * 2
        self.in_proj = nn.Linear(config.d_model, in_proj_size, bias=True)

        # Fused conv1d + SiLU (CUDA kernel) — on gate pathway
        self.conv = Conv1dSiLU(config.d_inner, kernel_size=config.conv_kernel_size)

        # Fused gate projection: all per-head scalars from x_conv in one GEMM
        # Layout: [alpha(H), omega(H), sel_B(H), sel_C(H), sel_dt(H), beta(H), r_gate(H)]
        self.gate_proj = nn.Linear(config.d_inner, config.n_heads * 7, bias=True)

        # Adaptive timestep: learnable per-head scale
        self.log_dt_scale = nn.Parameter(torch.zeros(config.n_heads))

        # SSD Scanner (Mamba-2 algorithm) with config-derived chunk_size
        self.ssd_scan = SSDChunkwiseScan(config.d_inner, config.n_heads, chunk_size=config.chunk_size)

        # Position encoding frequencies (RoPE-like, not learned)
        freqs = 1.0 / (config.rope_base ** (
            torch.arange(0, config.n_heads, dtype=torch.float32) / config.n_heads
        ))
        self.register_buffer("rope_freqs", freqs, persistent=False)

        # Q projection: D-dimensional query per head
        self.Q_proj = nn.Linear(config.d_inner, config.d_inner, bias=False)

        # Readout projection: maps retrieved 2-vectors back to feature space
        self.readout_proj = nn.Linear(config.n_heads * 2, config.d_inner, bias=False)

        # Output projection
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)

        # RMSNorm with config-derived epsilon (io_eps^2)
        self.norm = RMSNorm(config.d_model, eps=config.eps_norm)
        self.ssm_norm = nn.GroupNorm(config.ssm_norm_groups, config.d_inner)
        self.D = nn.Parameter(torch.zeros(config.d_inner))

        # Initialize weights
        self._init_weights()

        # Apply spectral initialization (uses config for band fraction, eps, etc.)
        apply_spectral_init(self, config=config, layer_idx=layer_idx)

    def _init_weights(self):
        """Variance-preserving initialization."""
        config = self.config
        stds = compute_variance_preserving_std(
            config.d_model, config.d_inner, config.n_layers,
        )

        nn.init.normal_(self.in_proj.weight, std=stds["in_proj"])
        nn.init.zeros_(self.in_proj.bias)

        with torch.no_grad():
            self.in_proj.bias[:config.d_inner].fill_(0.0)  # Z gate: 50% open
            # K: standard init, D-dimensional
            k_start = config.d_inner * 2
            k_end = k_start + config.d_inner
            self.in_proj.weight[k_start:k_end, :].normal_(std=stds["in_proj"])
            # V: n_heads * 2, small init
            v_start = k_end
            v_end = v_start + config.n_heads * 2
            self.in_proj.weight[v_start:v_end, :].normal_(
                std=stds["in_proj"] / math.sqrt(config.head_dim)
            )

        # Fused gate_proj initialization: per-segment stds and biases
        H = config.n_heads
        selection_std = 1.0 / math.sqrt(config.d_inner)
        dyn_std = stds["dynamics_proj"]
        biases = config.init_biases

        with torch.no_grad():
            nn.init.normal_(self.gate_proj.weight[:2*H], std=dyn_std)
            nn.init.normal_(self.gate_proj.weight[2*H:], std=selection_std)

            nn.init.zeros_(self.gate_proj.bias[:2*H])
            nn.init.constant_(self.gate_proj.bias[2*H:3*H], biases["sel_B_bias"])
            nn.init.constant_(self.gate_proj.bias[3*H:4*H], biases["sel_C_bias"])
            nn.init.constant_(self.gate_proj.bias[4*H:5*H], biases["sel_dt_bias"])
            nn.init.constant_(self.gate_proj.bias[5*H:6*H], biases["beta_bias"])
            nn.init.constant_(self.gate_proj.bias[6*H:7*H], biases["r_gate_bias"])

        nn.init.eye_(self.Q_proj.weight)
        nn.init.normal_(self.readout_proj.weight, std=1.0 / math.sqrt(config.n_heads * 2))
        nn.init.normal_(self.out_proj.weight, std=stds["out_proj"])
        nn.init.zeros_(self.D)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        config = self.config
        batch, seq_len, _ = x.shape
        residual = x

        x = self.norm(x)
        proj = self.in_proj(x)

        # Split into z_gate, x_gate, K, V
        z, x_gate, K, V = proj.split(
            [config.d_inner, config.d_inner, config.d_inner, config.n_heads * 2],
            dim=-1,
        )
        K = K.view(batch, seq_len, config.n_heads, config.head_dim)  # (B, L, H, D)
        V = V.view(batch, seq_len, config.n_heads, 2)                # (B, L, H, 2)

        # Conv on gate pathway (decoupled from V)
        x_conv = self.conv(x_gate)  # (B, L, d_inner)

        # Fused gate projection: all per-head scalars in one GEMM
        x_conv_cast = x_conv.to(self.gate_proj.weight.dtype)
        gates = self.gate_proj(x_conv_cast)  # (B, L, 7*H)

        # === Fused Dynamics Kernel ===
        A_bar, vp_scale, beta, sel_C_gate = dynamics_fused_cuda(
            gates, self.log_dt_scale.float(), self.rope_freqs,
            config.gating_c, config.bf16_omega_thresh,
            config.bf16_delta, config.bf16_smoothness, config.eps_adaptive_dt,
            config.n_heads,
        )

        # === Fused K/Q Normalization ===
        Q = self.Q_proj(x_conv_cast).view(batch, seq_len, config.n_heads, config.head_dim)
        K, Q = normalize_kq_cuda(K, Q)

        # Apply VP scale to V before scan
        V_gated = V * vp_scale.unsqueeze(-1)  # (B, L, H, 2)

        # Chunkwise Parallel Scan (CUDA kernels for intra/inter-chunk)
        Y, chunk_states = self.ssd_scan(A_bar, K, V_gated, beta)

        # Readout: retrieve 2-vectors from matrix state using Q
        retrieved = torch.einsum('blhsd,blhd->blhs', Y, Q)

        # Apply sel_C as read gate
        retrieved = retrieved * sel_C_gate.unsqueeze(-1)  # (B, L, H, 2)

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
        return residual + output
