"""Main KSSM Layer module.

The KSSMLayer is the core building block of the KSSM architecture.
It combines:
1. Input projections to generate α, ω, B, dt
2. Cayley transform discretization
3. State evolution (parallel scan or sequential recurrence)
4. Output projection

This layer can be used standalone or stacked in a backbone.
"""

import torch
import torch.nn as nn
from torch import Tensor

from kssm.config import KSSMConfig
from kssm.modules.projections import KSSMProjections, OutputProjection
from kssm.ops.cayley_op import cayley_transform, cayley_transform_no_grad
from kssm.ops.scan_op import evolution, evolution_no_grad, evolution_fused


class KSSMLayer(nn.Module):
    """KSSM Layer: Kinetic State Space Model layer.

    Implements the full KSSM computation:
    1. Project input x to dynamic parameters (α, ω, B, dt)
    2. Discretize using Cayley transform -> (A_bar, u_bar)
    3. Evolve state using parallel scan -> states
    4. Project states to output

    Supports both training mode (parallel scan) and inference mode
    (sequential recurrence with state caching).
    """

    def __init__(self, config: KSSMConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_inner = config.d_inner

        # Input projections
        self.projections = KSSMProjections(config)

        # Output projection
        self.output_proj = OutputProjection(config)

        # Numerical stability epsilon
        self.eps = 1e-6

    def forward(
        self,
        x: Tensor,
        state: Tensor | None = None,
        use_triton: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through KSSM layer.

        Args:
            x: Input tensor, shape (batch, seq, d_model).
            state: Optional initial state, shape (batch, d_inner, 2).
                   If None, starts from zero state.
            use_triton: Whether to use Triton kernels.

        Returns:
            output: Output tensor, shape (batch, seq, d_model).
            final_state: Final hidden state, shape (batch, d_inner, 2).
                         Can be passed to next call for sequential processing.
        """
        batch, seq_len, _ = x.shape

        # Step 1: Project input to dynamic parameters
        x_inner, alpha, omega, B, dt = self.projections(x)

        # Step 2+3: FUSED Cayley transform + Evolution
        # This eliminates the massive A_bar intermediate tensor by computing
        # the Cayley discretization on-the-fly inside the scan kernel.
        states = evolution_fused(
            alpha, omega, dt, B, x_inner,
            initial_state=state,
            eps=self.eps,
            use_triton=use_triton,
        )

        # Step 4: Output projection
        output = self.output_proj(states)

        # Extract final state for potential sequential processing
        final_state = states[:, -1, :, :]  # (batch, d_inner, 2)

        return output, final_state

    @torch.no_grad()
    def step(
        self,
        x: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Single step for autoregressive inference.

        Args:
            x: Input tensor, shape (batch, 1, d_model) or (batch, d_model).
            state: Current state, shape (batch, d_inner, 2).

        Returns:
            output: Output tensor, shape (batch, 1, d_model) or (batch, d_model).
            new_state: Updated state, shape (batch, d_inner, 2).
        """
        squeeze_output = x.dim() == 2
        if squeeze_output:
            x = x.unsqueeze(1)  # (batch, 1, d_model)

        batch = x.shape[0]

        # Project input
        x_inner, alpha, omega, B, dt = self.projections(x)

        # Cayley transform (single step)
        A_bar, u_bar = cayley_transform_no_grad(
            alpha, omega, dt, B, x_inner,
            eps=self.eps,
        )

        # Single step state update: h_new = A_bar @ h_old + u_bar
        # A_bar: (batch, 1, d_inner, 4), u_bar: (batch, 1, d_inner, 2)
        # state: (batch, d_inner, 2)

        A_bar = A_bar.squeeze(1).float()  # (batch, d_inner, 4)
        u_bar = u_bar.squeeze(1).float()  # (batch, d_inner, 2)
        state = state.float()

        a11 = A_bar[..., 0]
        a12 = A_bar[..., 1]
        a21 = A_bar[..., 2]
        a22 = A_bar[..., 3]

        h1 = state[..., 0]
        h2 = state[..., 1]

        new_h1 = a11 * h1 + a12 * h2 + u_bar[..., 0]
        new_h2 = a21 * h1 + a22 * h2 + u_bar[..., 1]

        new_state = torch.stack([new_h1, new_h2], dim=-1)  # (batch, d_inner, 2)

        # Output projection
        states = new_state.unsqueeze(1)  # (batch, 1, d_inner, 2)
        output = self.output_proj(states)  # (batch, 1, d_model)

        if squeeze_output:
            output = output.squeeze(1)

        return output, new_state.to(x.dtype)

    def init_state(self, batch_size: int, device: torch.device = None) -> Tensor:
        """Initialize hidden state to zeros.

        Args:
            batch_size: Batch size.
            device: Device for the state tensor.

        Returns:
            state: Zero state, shape (batch_size, d_inner, 2).
        """
        if device is None:
            device = next(self.parameters()).device

        return torch.zeros(
            batch_size, self.d_inner, 2,
            dtype=self.config.dtype,
            device=device,
        )


class KSSMLayerSimple(nn.Module):
    """Simplified KSSM Layer without input expansion.

    For testing and smaller models where d_inner == d_model.
    """

    def __init__(
        self,
        d_model: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Direct projections (no expansion)
        self.alpha_proj = nn.Linear(d_model, d_model)
        self.omega_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_model * 2)
        self.dt_proj = nn.Linear(d_model, d_model)

        # Output projection from 2D state
        self.C_proj = nn.Linear(d_model * 2, d_model)

        # Initialize dt bias for log-uniform distribution
        # Use stable inverse softplus: bias = dt + log(1 - exp(-dt))
        import math
        with torch.no_grad():
            log_dt_min = math.log(dt_min)
            log_dt_max = math.log(dt_max)
            log_dts = torch.linspace(log_dt_min, log_dt_max, d_model)
            dts = torch.exp(log_dts)
            # Inverse softplus: bias = dt + log(-expm1(-dt))
            biases = dts + torch.log(-torch.expm1(-dts))
            self.dt_proj.bias.copy_(biases)

    def forward(
        self,
        x: Tensor,
        state: Tensor | None = None,
        use_triton: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass."""
        import torch.nn.functional as F

        batch, seq_len, d_model = x.shape

        # Generate parameters
        alpha = F.softplus(self.alpha_proj(x))  # (batch, seq, d_model)
        omega = self.omega_proj(x)
        B = self.B_proj(x).view(batch, seq_len, d_model, 2)
        dt = F.softplus(self.dt_proj(x))

        # Cayley transform
        A_bar, u_bar = cayley_transform(alpha, omega, dt, B, x, self.eps, use_triton)

        # Evolution
        states = evolution(A_bar, u_bar, state, use_triton)

        # Output projection
        states_flat = states.view(batch, seq_len, d_model * 2)
        output = self.C_proj(states_flat)

        final_state = states[:, -1, :, :]

        return output, final_state

    @torch.no_grad()
    def step(
        self,
        x: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Single step for autoregressive inference.

        Args:
            x: Input tensor, shape (batch, d_model).
            state: Current state, shape (batch, d_model, 2).

        Returns:
            output: Output tensor, shape (batch, d_model).
            new_state: Updated state, shape (batch, d_model, 2).
        """
        import torch.nn.functional as F

        batch = x.shape[0]

        # Generate parameters for single step
        alpha = F.softplus(self.alpha_proj(x))  # (batch, d_model)
        omega = self.omega_proj(x)
        B = self.B_proj(x).view(batch, self.d_model, 2)
        dt = F.softplus(self.dt_proj(x))

        # Cayley transform (single step computation)
        tau = dt / 2.0

        # Compute M^{-1} components
        one_plus_tau_alpha = 1.0 + tau * alpha
        tau_omega = tau * omega
        det_M = one_plus_tau_alpha ** 2 + tau_omega ** 2
        inv_det = 1.0 / (det_M + self.eps)

        m11 = one_plus_tau_alpha * inv_det
        m12 = tau_omega * inv_det
        m21 = -tau_omega * inv_det
        m22 = one_plus_tau_alpha * inv_det

        # N components
        one_minus_tau_alpha = 1.0 - tau * alpha
        n11 = one_minus_tau_alpha
        n12 = tau_omega
        n21 = -tau_omega
        n22 = one_minus_tau_alpha

        # A_bar = M^{-1} @ N
        a11 = m11 * n11 + m12 * n21
        a12 = m11 * n12 + m12 * n22
        a21 = m21 * n11 + m22 * n21
        a22 = m21 * n12 + m22 * n22

        # u_bar = dt * M^{-1} @ B @ x
        Bx0 = B[..., 0] * x
        Bx1 = B[..., 1] * x
        u0 = dt * (m11 * Bx0 + m12 * Bx1)
        u1 = dt * (m21 * Bx0 + m22 * Bx1)

        # State update: new_h = A_bar @ h + u_bar
        h1 = state[..., 0]
        h2 = state[..., 1]

        new_h1 = a11 * h1 + a12 * h2 + u0
        new_h2 = a21 * h1 + a22 * h2 + u1

        new_state = torch.stack([new_h1, new_h2], dim=-1)  # (batch, d_model, 2)

        # Output projection
        states_flat = new_state.view(batch, self.d_model * 2)
        output = self.C_proj(states_flat)

        return output, new_state

    def init_state(self, batch_size: int, device: torch.device = None) -> Tensor:
        """Initialize hidden state."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.d_model, 2, device=device)
