"""Linear projections for KSSM parameters.

Generates the dynamic parameters α (damping), ω (frequency), B (input), C (output)
from the input sequence. These are then used to compute the Cayley discretization.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from kssm.config import KSSMConfig


class KSSMProjections(nn.Module):
    """Projects input to KSSM dynamic parameters with fused projection.

    Given input x of shape (batch, seq, d_model), produces:
    - alpha: Damping coefficients (batch, seq, d_inner), strictly positive via softplus
    - omega: Frequency coefficients (batch, seq, d_inner), unbounded
    - B: Input projection (batch, seq, d_inner, 2)
    - dt: Timestep (batch, seq, d_inner), positive via softplus

    Uses a single fused GEMM for all parameter projections (α, ω, B, dt),
    reducing kernel launch overhead similar to QKV fusion in attention.

    The output projection C is handled separately in the layer.
    """

    def __init__(self, config: KSSMConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_inner = config.d_inner

        # Input expansion projection
        self.in_proj = nn.Linear(config.d_model, config.d_inner)

        # Fused parameter projection: α + ω + B + dt
        # Output sizes: d_inner + d_inner + (d_inner * 2) + d_inner = 5 * d_inner
        self.kssm_proj = nn.Linear(config.d_inner, config.d_inner * 5)

        # Split sizes for torch.split
        self._split_sizes = [
            config.d_inner,      # alpha
            config.d_inner,      # omega
            config.d_inner * 2,  # B (2D state)
            config.d_inner,      # dt
        ]

        # Initialize weights and dt bias
        self._init_weights()

    def _init_weights(self):
        """Initialize fused projection weights and dt bias."""
        config = self.config
        d = self.d_inner

        # Standard initialization for most projections
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.kssm_proj.weight)

        if self.kssm_proj.bias is not None:
            # Zero bias for alpha, omega, B
            nn.init.zeros_(self.kssm_proj.bias[:d])       # alpha bias
            nn.init.zeros_(self.kssm_proj.bias[d:2*d])    # omega bias
            nn.init.zeros_(self.kssm_proj.bias[2*d:4*d])  # B bias

            # Log-uniform initialization for dt bias
            with torch.no_grad():
                log_dt_min = math.log(config.dt_min)
                log_dt_max = math.log(config.dt_max)
                log_dts = torch.linspace(log_dt_min, log_dt_max, d)
                dts = torch.exp(log_dts)
                # Inverse softplus: bias = dt + log(1 - exp(-dt))
                dt_biases = dts + torch.log(-torch.expm1(-dts))
                self.kssm_proj.bias[4*d:5*d].copy_(dt_biases)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Project input to KSSM parameters.

        Args:
            x: Input tensor, shape (batch, seq, d_model).

        Returns:
            x_inner: Expanded input, shape (batch, seq, d_inner).
            alpha: Damping coefficients, shape (batch, seq, d_inner). Strictly > 0.
            omega: Frequency coefficients, shape (batch, seq, d_inner).
            B: Input projection, shape (batch, seq, d_inner, 2).
            dt: Timestep, shape (batch, seq, d_inner). Strictly > 0.
        """
        batch, seq_len, _ = x.shape

        # Expand input dimension
        x_inner = self.in_proj(x)  # (batch, seq, d_inner)

        # Single fused GEMM for all parameters
        combined = self.kssm_proj(x_inner)  # (batch, seq, 5 * d_inner)

        # Split into individual parameters
        alpha_pre, omega, B_flat, dt_pre = torch.split(
            combined, self._split_sizes, dim=-1
        )

        # Apply activations
        alpha = F.softplus(alpha_pre)  # (batch, seq, d_inner), strictly > 0
        dt = F.softplus(dt_pre)        # (batch, seq, d_inner), strictly > 0

        # Reshape B to 2D state format
        B = B_flat.view(batch, seq_len, self.d_inner, 2)

        return x_inner, alpha, omega, B, dt


class KSSMProjectionsSeparate(nn.Module):
    """Projects input to KSSM dynamic parameters (separate projections).

    This is the non-fused reference implementation. Use KSSMProjections
    for better performance with fused GEMM.
    """

    def __init__(self, config: KSSMConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_inner = config.d_inner

        # Input expansion projection
        self.in_proj = nn.Linear(config.d_model, config.d_inner)

        # Separate parameter projections
        self.alpha_proj = nn.Linear(config.d_inner, config.d_inner)
        self.omega_proj = nn.Linear(config.d_inner, config.d_inner)
        self.B_proj = nn.Linear(config.d_inner, config.d_inner * 2)
        self.dt_proj = nn.Linear(config.d_inner, config.d_inner)

        # Initialize dt bias for log-uniform distribution
        self._init_dt_bias()

    def _init_dt_bias(self):
        """Initialize dt bias for log-uniform timestep distribution."""
        config = self.config

        with torch.no_grad():
            log_dt_min = math.log(config.dt_min)
            log_dt_max = math.log(config.dt_max)
            log_dts = torch.linspace(log_dt_min, log_dt_max, self.d_inner)
            dts = torch.exp(log_dts)
            biases = dts + torch.log(-torch.expm1(-dts))
            self.dt_proj.bias.copy_(biases)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Project input to KSSM parameters."""
        batch, seq_len, _ = x.shape

        x_inner = self.in_proj(x)

        alpha = F.softplus(self.alpha_proj(x_inner))
        omega = self.omega_proj(x_inner)
        B_flat = self.B_proj(x_inner)
        B = B_flat.view(batch, seq_len, self.d_inner, 2)
        dt = F.softplus(self.dt_proj(x_inner))

        return x_inner, alpha, omega, B, dt


class OutputProjection(nn.Module):
    """Output projection from 2D state back to model dimension.

    Projects the 2D hidden state back to d_model dimension.
    """

    def __init__(self, config: KSSMConfig):
        super().__init__()
        self.config = config

        # C: project 2D state to d_inner
        self.C_proj = nn.Linear(config.d_inner * 2, config.d_inner)

        # Output projection back to d_model
        self.out_proj = nn.Linear(config.d_inner, config.d_model)

    def forward(self, states: Tensor) -> Tensor:
        """Project states to output.

        Args:
            states: Hidden states, shape (batch, seq, d_inner, 2).

        Returns:
            output: Output tensor, shape (batch, seq, d_model).
        """
        batch, seq_len, d_inner, _ = states.shape

        # Flatten 2D state: (batch, seq, d_inner, 2) -> (batch, seq, d_inner * 2)
        states_flat = states.view(batch, seq_len, d_inner * 2)

        # Match dtype with weights for the linear projection
        weight_dtype = self.C_proj.weight.dtype
        if states_flat.dtype != weight_dtype:
            states_flat = states_flat.to(weight_dtype)

        # Project through C
        y = self.C_proj(states_flat)  # (batch, seq, d_inner)

        # Project to output dimension
        output = self.out_proj(y)  # (batch, seq, d_model)

        return output


class SimpleProjections(nn.Module):
    """Simplified projections without input expansion (fused).

    For testing and smaller models where d_inner == d_model.
    Uses fused GEMM for all parameter projections.
    """

    def __init__(self, d_model: int, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Fused projection: α + ω + B + dt
        # Output sizes: d_model + d_model + (d_model * 2) + d_model = 5 * d_model
        self.kssm_proj = nn.Linear(d_model, d_model * 5)

        # Split sizes for torch.split
        self._split_sizes = [d_model, d_model, d_model * 2, d_model]

        # Initialize weights and dt bias
        self._init_weights(dt_min, dt_max)

    def _init_weights(self, dt_min: float, dt_max: float):
        """Initialize fused projection weights and dt bias."""
        d = self.d_model

        nn.init.xavier_uniform_(self.kssm_proj.weight)

        if self.kssm_proj.bias is not None:
            # Zero bias for alpha, omega, B
            nn.init.zeros_(self.kssm_proj.bias[:d])       # alpha bias
            nn.init.zeros_(self.kssm_proj.bias[d:2*d])    # omega bias
            nn.init.zeros_(self.kssm_proj.bias[2*d:4*d])  # B bias

            # Log-uniform initialization for dt bias
            with torch.no_grad():
                log_dt_min = math.log(dt_min)
                log_dt_max = math.log(dt_max)
                log_dts = torch.linspace(log_dt_min, log_dt_max, d)
                dts = torch.exp(log_dts)
                dt_biases = dts + torch.log(-torch.expm1(-dts))
                self.kssm_proj.bias[4*d:5*d].copy_(dt_biases)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Project input to parameters (no expansion).

        Args:
            x: Input tensor, shape (batch, seq, d_model).

        Returns:
            alpha, omega, B, dt
        """
        batch, seq_len, d_model = x.shape

        # Single fused GEMM
        combined = self.kssm_proj(x)  # (batch, seq, 5 * d_model)

        # Split into individual parameters
        alpha_pre, omega, B_flat, dt_pre = torch.split(
            combined, self._split_sizes, dim=-1
        )

        # Apply activations
        alpha = F.softplus(alpha_pre)
        dt = F.softplus(dt_pre)

        # Reshape B
        B = B_flat.view(batch, seq_len, d_model, 2)

        return alpha, omega, B, dt


class SimpleProjectionsSeparate(nn.Module):
    """Simplified projections without input expansion (separate, for reference).

    This is the non-fused reference implementation.
    """

    def __init__(self, d_model: int, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Separate projections
        self.alpha_proj = nn.Linear(d_model, d_model)
        self.omega_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_model * 2)
        self.dt_proj = nn.Linear(d_model, d_model)

        # Initialize dt bias
        with torch.no_grad():
            log_dt_min = math.log(dt_min)
            log_dt_max = math.log(dt_max)
            log_dts = torch.linspace(log_dt_min, log_dt_max, d_model)
            dts = torch.exp(log_dts)
            biases = dts + torch.log(-torch.expm1(-dts))
            self.dt_proj.bias.copy_(biases)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Project input to parameters."""
        batch, seq_len, d_model = x.shape

        alpha = F.softplus(self.alpha_proj(x))
        omega = self.omega_proj(x)
        B = self.B_proj(x).view(batch, seq_len, d_model, 2)
        dt = F.softplus(self.dt_proj(x))

        return alpha, omega, B, dt
