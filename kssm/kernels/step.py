"""Triton kernel for single-step KSSM inference.

This kernel performs O(1) state update for autoregressive generation:
    h_t = A_bar @ h_{t-1} + u_bar

The key insight is that for generation, we only need to compute one timestep
at a time, which is a simple 2x2 matrix-vector multiply per feature dimension.

This is much more efficient than using the full parallel scan kernel with L=1
due to reduced kernel launch overhead.
"""

import triton
import triton.language as tl
import torch
from torch import Tensor


@triton.jit
def step_kernel(
    # Inputs
    A_bar_ptr,      # (batch, d_inner, 4) - discretized transition matrices
    u_bar_ptr,      # (batch, d_inner, 2) - discretized input
    state_ptr,      # (batch, d_inner, 2) - previous state (h_{t-1})
    # Output
    new_state_ptr,  # (batch, d_inner, 2) - new state (h_t)
    # Dimensions
    batch: tl.constexpr,
    d_inner: tl.constexpr,
    # Strides for A_bar
    stride_ab: tl.constexpr,
    stride_ad: tl.constexpr,
    stride_a4: tl.constexpr,
    # Strides for u_bar
    stride_ub: tl.constexpr,
    stride_ud: tl.constexpr,
    stride_u2: tl.constexpr,
    # Strides for state
    stride_sb: tl.constexpr,
    stride_sd: tl.constexpr,
    stride_s2: tl.constexpr,
    # Block size
    BLOCK_D: tl.constexpr,
):
    """Single-step state update kernel.

    Computes: h_t = A_bar @ h_{t-1} + u_bar

    Grid: (batch, cdiv(d_inner, BLOCK_D))
    """
    pid_b = tl.program_id(0)  # Batch index
    pid_d = tl.program_id(1)  # Feature block index

    # Feature dimension offsets
    d_offset = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offset < d_inner

    # Load A_bar components
    A_base = A_bar_ptr + pid_b * stride_ab + d_offset * stride_ad
    a11 = tl.load(A_base + 0 * stride_a4, mask=d_mask, other=0.0).to(tl.float32)
    a12 = tl.load(A_base + 1 * stride_a4, mask=d_mask, other=0.0).to(tl.float32)
    a21 = tl.load(A_base + 2 * stride_a4, mask=d_mask, other=0.0).to(tl.float32)
    a22 = tl.load(A_base + 3 * stride_a4, mask=d_mask, other=0.0).to(tl.float32)

    # Load u_bar
    u_base = u_bar_ptr + pid_b * stride_ub + d_offset * stride_ud
    u1 = tl.load(u_base + 0 * stride_u2, mask=d_mask, other=0.0).to(tl.float32)
    u2 = tl.load(u_base + 1 * stride_u2, mask=d_mask, other=0.0).to(tl.float32)

    # Load previous state
    s_base = state_ptr + pid_b * stride_sb + d_offset * stride_sd
    h1 = tl.load(s_base + 0 * stride_s2, mask=d_mask, other=0.0).to(tl.float32)
    h2 = tl.load(s_base + 1 * stride_s2, mask=d_mask, other=0.0).to(tl.float32)

    # Compute new state: h_t = A_bar @ h_{t-1} + u_bar
    new_h1 = a11 * h1 + a12 * h2 + u1
    new_h2 = a21 * h1 + a22 * h2 + u2

    # Store new state
    out_base = new_state_ptr + pid_b * stride_sb + d_offset * stride_sd
    tl.store(out_base + 0 * stride_s2, new_h1.to(tl.bfloat16), mask=d_mask)
    tl.store(out_base + 1 * stride_s2, new_h2.to(tl.bfloat16), mask=d_mask)


def kssm_step(
    A_bar: Tensor,
    u_bar: Tensor,
    state: Tensor,
    block_d: int = 64,
) -> Tensor:
    """Single-step KSSM state update.

    Computes h_t = A_bar @ h_{t-1} + u_bar for a single timestep.

    Args:
        A_bar: Discretized transition matrices, shape (batch, d_inner, 4).
               Must be bfloat16.
        u_bar: Discretized input, shape (batch, d_inner, 2).
               Must be bfloat16.
        state: Previous state h_{t-1}, shape (batch, d_inner, 2).
               Must be bfloat16.
        block_d: Block size for d_inner dimension.

    Returns:
        new_state: New state h_t, shape (batch, d_inner, 2), bfloat16.
    """
    # Ensure contiguous
    if not A_bar.is_contiguous():
        A_bar = A_bar.contiguous()
    if not u_bar.is_contiguous():
        u_bar = u_bar.contiguous()
    if not state.is_contiguous():
        state = state.contiguous()

    batch, d_inner, _ = A_bar.shape

    # Allocate output
    new_state = torch.empty(
        batch, d_inner, 2,
        dtype=torch.bfloat16,
        device=A_bar.device,
    )

    # Compute grid
    grid = (batch, triton.cdiv(d_inner, block_d))

    # Launch kernel
    step_kernel[grid](
        A_bar, u_bar, state, new_state,
        batch, d_inner,
        # A_bar strides
        A_bar.stride(0), A_bar.stride(1), A_bar.stride(2),
        # u_bar strides
        u_bar.stride(0), u_bar.stride(1), u_bar.stride(2),
        # state strides
        state.stride(0), state.stride(1), state.stride(2),
        # Block size
        BLOCK_D=block_d,
    )

    return new_state


@triton.jit
def step_with_cayley_kernel(
    # Inputs
    alpha_ptr,      # (batch, d_inner) - damping coefficients
    omega_ptr,      # (batch, d_inner) - frequency coefficients
    dt_ptr,         # (batch, d_inner) or scalar - timestep
    Bx_ptr,         # (batch, d_inner, 2) - B @ x (projected input)
    state_ptr,      # (batch, d_inner, 2) - previous state
    # Output
    new_state_ptr,  # (batch, d_inner, 2) - new state
    # Dimensions
    batch: tl.constexpr,
    d_inner: tl.constexpr,
    # Strides
    stride_ab: tl.constexpr,
    stride_ad: tl.constexpr,
    stride_bxb: tl.constexpr,
    stride_bxd: tl.constexpr,
    stride_bx2: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sd: tl.constexpr,
    stride_s2: tl.constexpr,
    # Config
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused Cayley + step kernel for inference.

    Computes discretization and state update in one kernel:
        A_bar = cayley(alpha, omega, dt)
        u_bar = dt * M^{-1} @ Bx
        h_t = A_bar @ h_{t-1} + u_bar

    Grid: (batch, cdiv(d_inner, BLOCK_D))
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offset = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offset < d_inner

    # Load parameters
    alpha_base = alpha_ptr + pid_b * stride_ab + d_offset * stride_ad
    alpha = tl.load(alpha_base, mask=d_mask, other=0.0).to(tl.float32)

    omega_base = omega_ptr + pid_b * stride_ab + d_offset * stride_ad
    omega = tl.load(omega_base, mask=d_mask, other=0.0).to(tl.float32)

    dt_base = dt_ptr + pid_b * stride_ab + d_offset * stride_ad
    dt = tl.load(dt_base, mask=d_mask, other=1.0).to(tl.float32)

    # Load Bx
    Bx_base = Bx_ptr + pid_b * stride_bxb + d_offset * stride_bxd
    Bx1 = tl.load(Bx_base + 0 * stride_bx2, mask=d_mask, other=0.0).to(tl.float32)
    Bx2 = tl.load(Bx_base + 1 * stride_bx2, mask=d_mask, other=0.0).to(tl.float32)

    # Load previous state
    s_base = state_ptr + pid_b * stride_sb + d_offset * stride_sd
    h1 = tl.load(s_base + 0 * stride_s2, mask=d_mask, other=0.0).to(tl.float32)
    h2 = tl.load(s_base + 1 * stride_s2, mask=d_mask, other=0.0).to(tl.float32)

    # Cayley transform
    tau = dt / 2.0
    det_M = (1.0 + tau * alpha) * (1.0 + tau * alpha) + (tau * omega) * (tau * omega)
    inv_det = 1.0 / (det_M + EPS)

    # M^{-1} components
    m11 = (1.0 + tau * alpha) * inv_det
    m12 = (tau * omega) * inv_det
    m21 = -(tau * omega) * inv_det
    m22 = (1.0 + tau * alpha) * inv_det

    # N = I + tau*A components
    n11 = 1.0 - tau * alpha
    n12 = tau * omega
    n21 = -tau * omega
    n22 = 1.0 - tau * alpha

    # A_bar = M^{-1} @ N
    a11 = m11 * n11 + m12 * n21
    a12 = m11 * n12 + m12 * n22
    a21 = m21 * n11 + m22 * n21
    a22 = m21 * n12 + m22 * n22

    # u_bar = dt * M^{-1} @ Bx
    u1 = dt * (m11 * Bx1 + m12 * Bx2)
    u2 = dt * (m21 * Bx1 + m22 * Bx2)

    # State update: h_t = A_bar @ h_{t-1} + u_bar
    new_h1 = a11 * h1 + a12 * h2 + u1
    new_h2 = a21 * h1 + a22 * h2 + u2

    # Store new state
    out_base = new_state_ptr + pid_b * stride_sb + d_offset * stride_sd
    tl.store(out_base + 0 * stride_s2, new_h1.to(tl.bfloat16), mask=d_mask)
    tl.store(out_base + 1 * stride_s2, new_h2.to(tl.bfloat16), mask=d_mask)


def kssm_step_fused(
    alpha: Tensor,
    omega: Tensor,
    dt: Tensor,
    Bx: Tensor,
    state: Tensor,
    eps: float = 1e-6,
    block_d: int = 64,
) -> Tensor:
    """Fused Cayley + step for inference.

    Combines discretization and state update in one kernel call.

    Args:
        alpha: Damping coefficients, shape (batch, d_inner). Must be >= 0.
        omega: Frequency coefficients, shape (batch, d_inner).
        dt: Timestep, shape (batch, d_inner).
        Bx: Projected input B @ x, shape (batch, d_inner, 2).
        state: Previous state, shape (batch, d_inner, 2).
        eps: Epsilon for numerical stability.
        block_d: Block size.

    Returns:
        new_state: New state h_t, shape (batch, d_inner, 2), bfloat16.
    """
    # Ensure contiguous and bf16
    if not alpha.is_contiguous():
        alpha = alpha.contiguous()
    if not omega.is_contiguous():
        omega = omega.contiguous()
    if not dt.is_contiguous():
        dt = dt.contiguous()
    if not Bx.is_contiguous():
        Bx = Bx.contiguous()
    if not state.is_contiguous():
        state = state.contiguous()

    # Convert to bf16 if needed
    if alpha.dtype != torch.bfloat16:
        alpha = alpha.bfloat16()
    if omega.dtype != torch.bfloat16:
        omega = omega.bfloat16()
    if dt.dtype != torch.bfloat16:
        dt = dt.bfloat16()
    if Bx.dtype != torch.bfloat16:
        Bx = Bx.bfloat16()
    if state.dtype != torch.bfloat16:
        state = state.bfloat16()

    batch, d_inner = alpha.shape

    # Allocate output
    new_state = torch.empty(
        batch, d_inner, 2,
        dtype=torch.bfloat16,
        device=alpha.device,
    )

    # Compute grid
    grid = (batch, triton.cdiv(d_inner, block_d))

    # Launch kernel
    step_with_cayley_kernel[grid](
        alpha, omega, dt, Bx, state, new_state,
        batch, d_inner,
        # alpha/omega/dt strides (same layout)
        alpha.stride(0), alpha.stride(1),
        # Bx strides
        Bx.stride(0), Bx.stride(1), Bx.stride(2),
        # state strides
        state.stride(0), state.stride(1), state.stride(2),
        # Config
        EPS=eps,
        BLOCK_D=block_d,
    )

    return new_state
