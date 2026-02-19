import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cdssm.config.model import CDSSMConfig
from cdssm.ops import DynamicsFusedFn, NormalizeKQFn
from cdssm.models.ssd import SSDChunkwiseScan
from cdssm.models.modules import Conv1dSiLU, RMSNorm
from cdssm.models.init import compute_variance_preserving_std, apply_spectral_init


class CDSSMBlock(nn.Module):
    """Cayley-stable dissipative SSM block with complex diagonal state.

    Uses N/2 independent complex eigenvalue pairs per head, discretized via the
    Cayley transform guaranteeing |A_bar_j| <= 1 (A-stability). State h in
    R^(N x D) stores N/2 complex pairs as re/im interleaved.

    Architecture:
    - Fused gate projection: N+5 per-head scalars in one GEMM
      Layout: [alpha_0..alpha_{N/2-1}, omega_0..omega_{N/2-1},
               sel_B, sel_C, sel_dt, beta, r_gate] x H
    - Matrix memory: state S in R^(N x D) per head for delta-rule retrieval
    - Per-eigenvalue VP scale and recurrence gate modulation
    - L2-normalized keys and queries
    - Value residual learning (RWKV-7 style cross-layer interpolation)
    - Current-token shortcut (direct Q*K*V bypass)
    - Position encoding via RoPE-like omega modulation

    All constants derived from CDSSMConfig.
    """

    def __init__(self, config: CDSSMConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        N = config.state_dim

        # in_proj: [z_gate, x_gate, K, V]
        in_proj_size = config.d_inner * 3 + config.n_heads * N
        self.in_proj = nn.Linear(config.d_model, in_proj_size, bias=True)

        # Fused conv1d + SiLU (CUDA kernel) — on gate pathway
        self.conv = Conv1dSiLU(config.d_inner, kernel_size=config.conv_kernel_size)

        # Fused gate projection: all per-head scalars from x_conv in one GEMM
        # Layout: [alpha(N/2*H), omega(N/2*H), sel_B(H), sel_C(H), sel_dt(H), beta(H), r_gate(H)]
        G = N + 5  # gates per head
        self.gate_proj = nn.Linear(config.d_inner, config.n_heads * G, bias=True)

        # Adaptive timestep: learnable per-head scale
        self.log_dt_scale = nn.Parameter(torch.zeros(config.n_heads))

        # SSD Scanner (Mamba-2 algorithm) with config-derived chunk_size
        self.ssd_scan = SSDChunkwiseScan(config.chunk_size, config.state_dim)

        # Position encoding frequencies (RoPE-like, not learned)
        freqs = 1.0 / (config.rope_base ** (
            torch.arange(0, config.n_heads, dtype=torch.float32) / config.n_heads
        ))
        self.register_buffer("rope_freqs", freqs, persistent=False)

        # Q projection: D-dimensional query per head
        self.Q_proj = nn.Linear(config.d_inner, config.d_inner, bias=False)

        # Readout projection: maps retrieved N-vectors back to feature space
        self.readout_proj = nn.Linear(config.n_heads * N, config.d_inner, bias=False)

        # Output projection
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)

        # RMSNorm with config-derived epsilon (io_eps^2)
        self.norm = RMSNorm(config.d_model, eps=config.eps_norm)
        self.ssm_norm = nn.GroupNorm(config.ssm_norm_groups, config.d_inner)
        self.D = nn.Parameter(torch.zeros(config.d_inner))

        # Value residual learning (RWKV-7 style)
        self.v_residual_gate = nn.Parameter(torch.zeros(config.n_heads))

        # Current-token shortcut (initialized to 0 = off at init)
        self.shortcut_scale = nn.Parameter(torch.zeros(1))

        # Initialize weights
        self._init_weights()

        # Apply spectral initialization
        apply_spectral_init(self, config=config, layer_idx=layer_idx)

    def _init_weights(self):
        """Variance-preserving initialization."""
        config = self.config
        N = config.state_dim
        N_half = N // 2
        H = config.n_heads
        G = N + 5

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
            # V: n_heads * N, small init
            v_start = k_end
            v_end = v_start + config.n_heads * N
            self.in_proj.weight[v_start:v_end, :].normal_(
                std=stds["in_proj"] / math.sqrt(config.head_dim)
            )

        # Fused gate_proj initialization: per-segment stds and biases
        selection_std = 1.0 / math.sqrt(config.d_inner)
        dyn_std = stds["dynamics_proj"]
        biases = config.init_biases

        with torch.no_grad():
            # Alpha and omega segments (N_half * H each)
            nn.init.normal_(self.gate_proj.weight[:N * H], std=dyn_std)
            nn.init.normal_(self.gate_proj.weight[N * H:], std=selection_std)

            nn.init.zeros_(self.gate_proj.bias[:N * H])  # alpha + omega biases set by spectral init
            nn.init.constant_(self.gate_proj.bias[N * H:(N + 1) * H], biases["sel_B_bias"])
            nn.init.constant_(self.gate_proj.bias[(N + 1) * H:(N + 2) * H], biases["sel_C_bias"])
            nn.init.constant_(self.gate_proj.bias[(N + 2) * H:(N + 3) * H], biases["sel_dt_bias"])
            nn.init.constant_(self.gate_proj.bias[(N + 3) * H:(N + 4) * H], biases["beta_bias"])
            nn.init.constant_(self.gate_proj.bias[(N + 4) * H:(N + 5) * H], biases["r_gate_bias"])

        nn.init.eye_(self.Q_proj.weight)
        nn.init.normal_(self.readout_proj.weight, std=1.0 / math.sqrt(config.n_heads * N))
        nn.init.normal_(self.out_proj.weight, std=stds["out_proj"])
        nn.init.zeros_(self.D)

    def forward(self, x: Tensor, v_first: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: (B, L, d_model) input
            v_first: (B, L, H, N) layer-0 V for value residual, or None for layer 0

        Returns:
            (output, v_first): output tensor and v_first for next layer
        """
        config = self.config
        N = config.state_dim
        batch, seq_len, _ = x.shape
        residual = x

        x = self.norm(x)
        proj = self.in_proj(x)

        # Split into z_gate, x_gate, K, V
        z, x_gate, K, V = proj.split(
            [config.d_inner, config.d_inner, config.d_inner, config.n_heads * N],
            dim=-1,
        )
        K = K.view(batch, seq_len, config.n_heads, config.head_dim)  # (B, L, H, D)
        V = V.view(batch, seq_len, config.n_heads, N)                # (B, L, H, N)

        # Value residual learning
        if v_first is None:
            v_first = V.detach()  # Layer 0 captures initial values
        else:
            nu = torch.sigmoid(self.v_residual_gate)  # (H,)
            V = torch.lerp(v_first, V, nu[None, None, :, None])  # (B, L, H, N)

        # Conv on gate pathway (decoupled from V)
        x_conv = self.conv(x_gate)  # (B, L, d_inner)

        # Fused gate projection: all per-head scalars in one GEMM
        x_conv_cast = x_conv.to(self.gate_proj.weight.dtype)
        gates = self.gate_proj(x_conv_cast)  # (B, L, (N+5)*H)

        # === Fused Dynamics Kernel ===
        A_bar, vp_scale, beta, sel_C_gate = DynamicsFusedFn.apply(
            gates, self.log_dt_scale.float(), self.rope_freqs,
            config.gating_c, config.bf16_omega_thresh,
            config.bf16_delta, config.bf16_smoothness, config.eps_adaptive_dt,
            config.n_heads, config.state_dim,
        )

        # === Fused K/Q Normalization ===
        Q = self.Q_proj(x_conv_cast).view(batch, seq_len, config.n_heads, config.head_dim)
        K, Q = NormalizeKQFn.apply(K, Q)

        # Apply VP scale to V before scan
        # vp_scale: (B, L, H, N/2) — one per eigenvalue pair, broadcast to both re/im
        vp_scale_expanded = vp_scale.repeat_interleave(2, dim=-1)  # (B, L, H, N)
        V_gated = V * vp_scale_expanded

        # Chunkwise Parallel Scan (CUDA kernels for intra/inter-chunk)
        Y = self.ssd_scan(A_bar, K, V_gated, beta)  # (B, L, H, N, D)

        # Current-token shortcut: direct Q*K*V bypass around recurrent state
        # qk: (B, L, H) — dot product of Q and K per head
        qk = torch.einsum('blhd,blhd->blh', Q, K)
        # shortcut: (B, L, H, N, D) — V outer K scaled by qk
        shortcut = qk[..., None, None] * V_gated.unsqueeze(-1) * K.unsqueeze(-2)
        Y = Y + self.shortcut_scale * shortcut

        # Readout: retrieve N-vectors from matrix state using Q
        retrieved = torch.einsum('blhsd,blhd->blhs', Y, Q)

        # Apply sel_C as read gate
        retrieved = retrieved * sel_C_gate.unsqueeze(-1)  # (B, L, H, N)

        # Project retrieved N-vectors back to feature space
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
        return residual + output, v_first
