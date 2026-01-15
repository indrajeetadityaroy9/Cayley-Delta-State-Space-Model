"""Unified KSSM Block (Mamba-style architecture).

This module implements a unified gated SSM block that matches Mamba's architecture:
- Single in-projection (1 GEMM)
- Conv1d for local smoothing
- KSSM scan (fused Cayley + evolution)
- Gating (replaces separate MLP)
- Single out-projection (1 GEMM)

Key optimizations vs naive KSSM:
- Grouped B projection (like Mamba's d_state): reduces in_proj from 4.7M to 2.4M params
- Element-wise C output (like Mamba): eliminates 4.7M C_proj params
- Fused Cayley+Scan kernel: eliminates A_bar intermediate tensor
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from kssm.config import KSSMConfig
from kssm.ops.scan_op import evolution_fused


class KSSMBlock(nn.Module):
    """Unified KSSM Block (Mamba-style, parameter-efficient).

    Architecture:
        1. In-Projection: x -> [z (gate), x_ssm, B_groups, dt_params]
        2. Conv1d: Local smoothing on x_ssm
        3. KSSM Scan: Apply Cayley discretization + evolution
        4. Element-wise C output (no C_proj!)
        5. Gating: y = scan_out * SiLU(z)
        6. Out-Projection: y -> output

    Parameter efficiency (vs naive KSSM):
        - B is grouped (n_B_groups=d_state) and broadcast, not per-channel
        - C is element-wise weights, not a full projection
        - Total: ~3.8M params/block (matches Mamba) vs ~10.9M naive
    """

    def __init__(self, config: KSSMConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.d_conv = config.d_conv
        self.dt_rank = config.dt_rank
        self.d_state = config.d_state  # Used for grouped B (like Mamba)
        self.eps = 1e-6

        # Number of B groups (like Mamba's d_state)
        self.n_B_groups = self.d_state

        # ============================================================
        # 1. Input Projection (Fused - matches Mamba's param count)
        # ============================================================
        # Projects to: z (gate), x (ssm input), B_groups, dt_params
        # z: d_inner (for gating)
        # x: d_inner (SSM input)
        # B_groups: n_B_groups * 2 (grouped 2D state input, broadcast to d_inner)
        # dt_params: dt_rank (for alpha, omega, dt projection)
        in_proj_size = self.d_inner * 2 + self.n_B_groups * 2 + self.dt_rank
        self.in_proj = nn.Linear(self.d_model, in_proj_size, bias=False)

        # ============================================================
        # 2. Conv1d for Local Smoothing (Mamba-style)
        # ============================================================
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,  # Depthwise
            padding=self.d_conv - 1,  # Causal padding
            bias=True,
        )

        # ============================================================
        # 3. KSSM Parameter Projections (from dt_rank to d_inner)
        # ============================================================
        self.alpha_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.omega_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # ============================================================
        # 4. Element-wise C (like Mamba - NO full projection!)
        # ============================================================
        # C_weight: (d_inner, 2) - element-wise combination of 2D state
        # Output: y = sum(states * C_weight, dim=-1)
        # This saves 4.7M parameters vs C_proj!
        self.C_weight = nn.Parameter(torch.ones(self.d_inner, 2))

        # ============================================================
        # 5. Output Projection (back to d_model)
        # ============================================================
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # ============================================================
        # 6. LayerNorm (Pre-norm)
        # ============================================================
        self.norm = nn.LayerNorm(self.d_model)

        # D parameter (like Mamba's skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following Mamba conventions."""
        # In projection: small init
        nn.init.normal_(self.in_proj.weight, std=0.02)

        # Conv1d: small init
        nn.init.normal_(self.conv1d.weight, std=0.02)
        nn.init.zeros_(self.conv1d.bias)

        # Alpha projection: init for low damping (long memory)
        nn.init.zeros_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, -5.0)  # softplus(-5) ≈ 0.007

        # Omega projection: log-uniform frequencies
        nn.init.zeros_(self.omega_proj.weight)
        with torch.no_grad():
            freqs = torch.exp(torch.linspace(math.log(0.01), math.log(100.0), self.d_inner))
            self.omega_proj.bias.copy_(freqs)

        # dt projection: log-uniform timesteps
        nn.init.normal_(self.dt_proj.weight, std=0.02)
        with torch.no_grad():
            log_dt_min = math.log(self.config.dt_min)
            log_dt_max = math.log(self.config.dt_max)
            log_dts = torch.linspace(log_dt_min, log_dt_max, self.d_inner)
            dts = torch.exp(log_dts)
            dt_biases = dts + torch.log(-torch.expm1(-dts))
            self.dt_proj.bias.copy_(dt_biases)

        # C_weight: init to [1, 0] to read first state component by default
        with torch.no_grad():
            self.C_weight[:, 0] = 1.0
            self.C_weight[:, 1] = 0.0

        # Output projection
        nn.init.normal_(self.out_proj.weight, std=0.02)

        # D: ones (skip connection)
        nn.init.ones_(self.D)

    def forward(
        self,
        x: Tensor,
        state: Tensor | None = None,
        use_triton: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through unified block.

        Args:
            x: Input tensor, shape (batch, seq, d_model).
            state: Optional initial state, shape (batch, d_inner, 2).
            use_triton: Whether to use Triton kernels.

        Returns:
            output: Output tensor, shape (batch, seq, d_model).
            final_state: Final state, shape (batch, d_inner, 2).
        """
        batch, seq_len, _ = x.shape
        residual = x

        # Pre-norm
        x = self.norm(x)

        # ============================================================
        # A. Input Projection (Fused GEMM)
        # ============================================================
        proj = self.in_proj(x)  # (batch, seq, in_proj_size)

        z, x_ssm, B_groups, dt_params = proj.split(
            [self.d_inner, self.d_inner, self.n_B_groups * 2, self.dt_rank],
            dim=-1
        )
        # z: (batch, seq, d_inner) - gate branch
        # x_ssm: (batch, seq, d_inner) - SSM input branch
        # B_groups: (batch, seq, n_B_groups * 2) - grouped input matrix
        # dt_params: (batch, seq, dt_rank) - compressed params

        # ============================================================
        # B. Conv1d for Local Smoothing
        # ============================================================
        x_conv = x_ssm.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_ssm_act = F.silu(x_conv)

        # ============================================================
        # C. KSSM Parameter Generation
        # ============================================================
        alpha = F.softplus(self.alpha_proj(dt_params))
        omega = self.omega_proj(dt_params)
        dt = F.softplus(self.dt_proj(dt_params))

        # Expand B from groups to full d_inner (broadcast like Mamba)
        # B_groups: (batch, seq, n_B_groups, 2)
        B_groups = B_groups.view(batch, seq_len, self.n_B_groups, 2)
        # Repeat to match d_inner: each group covers d_inner // n_B_groups channels
        channels_per_group = self.d_inner // self.n_B_groups
        B = B_groups.repeat_interleave(channels_per_group, dim=2)  # (batch, seq, d_inner, 2)

        # ============================================================
        # D. KSSM Scan (Fused Cayley + Evolution)
        # ============================================================
        states = evolution_fused(
            alpha, omega, dt, B, x_ssm_act,
            initial_state=state,
            eps=self.eps,
            use_triton=use_triton,
        )  # (batch, seq, d_inner, 2)

        # ============================================================
        # E. Element-wise C Output (no projection!)
        # ============================================================
        # y = sum(states * C_weight, dim=-1) + D * x_ssm
        # Match dtype
        weight_dtype = self.C_weight.dtype
        if states.dtype != weight_dtype:
            states = states.to(weight_dtype)

        y = (states * self.C_weight).sum(dim=-1)  # (batch, seq, d_inner)
        y = y + self.D * x_ssm_act  # Skip connection (like Mamba's D)

        # ============================================================
        # F. Gating (Replaces MLP!)
        # ============================================================
        y = y * F.silu(z)

        # ============================================================
        # G. Output Projection + Residual
        # ============================================================
        output = self.out_proj(y)
        output = residual + output

        # Extract final state
        final_state = states[:, -1, :, :]

        return output, final_state

    @torch.no_grad()
    def step(
        self,
        x: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Single step for autoregressive inference."""
        batch = x.shape[0]
        residual = x

        x = self.norm(x)

        proj = self.in_proj(x)
        z, x_ssm, B_groups, dt_params = proj.split(
            [self.d_inner, self.d_inner, self.n_B_groups * 2, self.dt_rank],
            dim=-1
        )

        x_ssm_act = F.silu(x_ssm)

        alpha = F.softplus(self.alpha_proj(dt_params))
        omega = self.omega_proj(dt_params)
        dt = F.softplus(self.dt_proj(dt_params))

        # Expand B
        B_groups = B_groups.view(batch, self.n_B_groups, 2)
        channels_per_group = self.d_inner // self.n_B_groups
        B = B_groups.repeat_interleave(channels_per_group, dim=1)

        # Single-step KSSM update
        tau = dt / 2.0
        one_plus_tau_alpha = 1.0 + tau * alpha
        tau_omega = tau * omega
        det_M = one_plus_tau_alpha ** 2 + tau_omega ** 2
        inv_det = 1.0 / (det_M + self.eps)

        m11 = one_plus_tau_alpha * inv_det
        m12 = tau_omega * inv_det
        m21 = -tau_omega * inv_det
        m22 = one_plus_tau_alpha * inv_det

        one_minus_tau_alpha = 1.0 - tau * alpha
        a11 = m11 * one_minus_tau_alpha + m12 * (-tau_omega)
        a12 = m11 * tau_omega + m12 * one_minus_tau_alpha
        a21 = m21 * one_minus_tau_alpha + m22 * (-tau_omega)
        a22 = m21 * tau_omega + m22 * one_minus_tau_alpha

        Bx0 = B[..., 0] * x_ssm_act
        Bx1 = B[..., 1] * x_ssm_act
        u0 = dt * (m11 * Bx0 + m12 * Bx1)
        u1 = dt * (m21 * Bx0 + m22 * Bx1)

        h1 = state[..., 0]
        h2 = state[..., 1]
        new_h1 = a11 * h1 + a12 * h2 + u0
        new_h2 = a21 * h1 + a22 * h2 + u1
        new_state = torch.stack([new_h1, new_h2], dim=-1)

        # Element-wise C output
        y = (new_state * self.C_weight).sum(dim=-1)
        y = y + self.D * x_ssm_act

        y = y * F.silu(z)
        output = self.out_proj(y)
        output = residual + output

        return output, new_state

    def init_state(self, batch_size: int, device: torch.device = None) -> Tensor:
        """Initialize state to zeros."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.d_inner, 2, device=device)


# ============================================================
# Legacy classes for backward compatibility
# ============================================================

class GatedMLP(nn.Module):
    """Gated MLP (SwiGLU-style) - LEGACY, kept for compatibility."""

    def __init__(self, d_model: int, expand: int = 2, bias: bool = False):
        super().__init__()
        d_hidden = d_model * expand
        self.gate_proj = nn.Linear(d_model, d_hidden, bias=bias)
        self.up_proj = nn.Linear(d_model, d_hidden, bias=bias)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=bias)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class KSSMBlockSimple(nn.Module):
    """Simplified KSSM Block - LEGACY, kept for compatibility."""

    def __init__(
        self,
        d_model: int,
        mlp_expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        from kssm.modules.kssm_layer import KSSMLayerSimple

        self.norm_mixer = nn.LayerNorm(d_model)
        self.mixer = KSSMLayerSimple(d_model, dt_min, dt_max)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = GatedMLP(d_model, expand=mlp_expand)

    def forward(
        self,
        x: Tensor,
        state: Tensor | None = None,
        use_triton: bool = True,
    ) -> tuple[Tensor, Tensor]:
        residual = x
        mixer_out, final_state = self.mixer(self.norm_mixer(x), state, use_triton)
        x = residual + mixer_out

        residual = x
        x = residual + self.mlp(self.norm_mlp(x))

        return x, final_state

    @torch.no_grad()
    def step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        residual = x
        mixer_out, new_state = self.mixer.step(self.norm_mixer(x), state)
        x = residual + mixer_out

        residual = x
        x = residual + self.mlp(self.norm_mlp(x))

        return x, new_state

    def init_state(self, batch_size: int, device: torch.device = None) -> Tensor:
        return self.mixer.init_state(batch_size, device)
