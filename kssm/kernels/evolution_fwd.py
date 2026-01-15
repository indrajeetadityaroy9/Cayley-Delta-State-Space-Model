"""Fused Triton kernel for Cayley transform + Evolution (persistent scan).

This kernel FUSES the Cayley discretization directly into the scan loop,
eliminating the massive A_bar intermediate tensor from HBM entirely.

Key insight: A_bar (batch × seq × d_inner × 4) is never materialized.
Instead, we compute the 2x2 rotation matrix on-the-fly in registers.

Memory savings at L=8192, d_inner=1536, batch=4:
- Before: A_bar = 4 × 8192 × 1536 × 4 × 2 bytes = ~400 MB
- After: 0 bytes (computed in registers)

This is the "Big Win" optimization that matches Mamba's fused kernel approach.
"""

import triton
import triton.language as tl
import torch
from torch import Tensor


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 32}, num_warps=1),
        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ],
    key=["d_inner"],
)
@triton.jit
def evolution_fused_fwd_kernel(
    # Inputs (raw projections, NOT discretized matrices)
    alpha_ptr,  # (batch, seq, d_inner) - damping coefficients (post-softplus)
    omega_ptr,  # (batch, seq, d_inner) - frequency coefficients
    dt_ptr,     # (batch, seq, d_inner) - timestep (post-softplus)
    B_ptr,      # (batch, seq, d_inner, 2) - input projection
    x_ptr,      # (batch, seq, d_inner) - input values (expanded)
    # Optional initial state
    initial_state_ptr,  # (batch, d_inner, 2) or dummy
    # Outputs
    states_ptr,  # (batch, seq, d_inner, 2) - output states
    # Dimensions
    batch: tl.constexpr,
    seq_len: tl.constexpr,
    d_inner: tl.constexpr,
    has_initial: tl.constexpr,
    # Strides for alpha/omega/dt/x (batch, seq, d_inner)
    stride_pb: tl.constexpr,
    stride_ps: tl.constexpr,
    stride_pd: tl.constexpr,
    # Strides for B (batch, seq, d_inner, 2)
    stride_Bb: tl.constexpr,
    stride_Bs: tl.constexpr,
    stride_Bd: tl.constexpr,
    stride_B2: tl.constexpr,
    # Strides for initial_state (batch, d_inner, 2)
    stride_ib: tl.constexpr,
    stride_id: tl.constexpr,
    stride_i2: tl.constexpr,
    # Strides for states (batch, seq, d_inner, 2)
    stride_sb: tl.constexpr,
    stride_ss: tl.constexpr,
    stride_sd: tl.constexpr,
    stride_s2: tl.constexpr,
    # Numerical stability
    eps: tl.constexpr,
    # Block size
    BLOCK_D: tl.constexpr,
):
    """Fused Cayley + Evolution kernel.

    Grid: (batch, cdiv(d_inner, BLOCK_D))
    Each program handles one batch element and BLOCK_D features,
    looping over the entire sequence while keeping state in registers.

    Key optimization: A_bar is computed on-the-fly and never written to HBM.
    """
    pid_b = tl.program_id(0)  # Batch index
    pid_d = tl.program_id(1)  # Feature block index

    # Feature dimension offsets
    d_offset = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offset < d_inner

    # Initialize state in fp32 registers
    if has_initial:
        init_base = initial_state_ptr + pid_b * stride_ib + d_offset * stride_id
        h1 = tl.load(init_base + 0 * stride_i2, mask=d_mask, other=0.0).to(tl.float32)
        h2 = tl.load(init_base + 1 * stride_i2, mask=d_mask, other=0.0).to(tl.float32)
    else:
        h1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        h2 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Loop over sequence - this is the persistent scan
    for t in range(seq_len):
        # Compute base offsets for this timestep
        param_base = pid_b * stride_pb + t * stride_ps + d_offset * stride_pd
        B_base = pid_b * stride_Bb + t * stride_Bs + d_offset * stride_Bd
        s_base = states_ptr + pid_b * stride_sb + t * stride_ss + d_offset * stride_sd

        # ============================================================
        # FUSED CAYLEY TRANSFORM (computed in registers, never stored)
        # ============================================================

        # Load raw parameters (bf16 -> fp32)
        alpha = tl.load(alpha_ptr + param_base, mask=d_mask, other=0.0).to(tl.float32)
        omega = tl.load(omega_ptr + param_base, mask=d_mask, other=0.0).to(tl.float32)
        dt = tl.load(dt_ptr + param_base, mask=d_mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + param_base, mask=d_mask, other=0.0).to(tl.float32)

        # Load B (2 components)
        B0 = tl.load(B_ptr + B_base + 0 * stride_B2, mask=d_mask, other=0.0).to(tl.float32)
        B1 = tl.load(B_ptr + B_base + 1 * stride_B2, mask=d_mask, other=0.0).to(tl.float32)

        # Cayley transform: A_bar = (I - τA)^{-1}(I + τA), τ = dt/2
        tau = dt * 0.5

        # M = I - τA where A = [[-α, ω], [-ω, -α]]
        # M = [[1 + τα, -τω], [τω, 1 + τα]]
        # det(M) = (1 + τα)² + (τω)²
        one_plus_tau_alpha = 1.0 + tau * alpha
        tau_omega = tau * omega
        det_M = one_plus_tau_alpha * one_plus_tau_alpha + tau_omega * tau_omega
        inv_det = 1.0 / (det_M + eps)

        # M^{-1} = (1/det) * [[1 + τα, τω], [-τω, 1 + τα]]
        m11 = one_plus_tau_alpha * inv_det
        m12 = tau_omega * inv_det
        m21 = -tau_omega * inv_det
        m22 = one_plus_tau_alpha * inv_det

        # N = I + τA = [[1 - τα, τω], [-τω, 1 - τα]]
        one_minus_tau_alpha = 1.0 - tau * alpha
        n11 = one_minus_tau_alpha
        n12 = tau_omega
        n21 = -tau_omega
        n22 = one_minus_tau_alpha

        # A_bar = M^{-1} @ N (2x2 matrix multiply, stays in registers!)
        a11 = m11 * n11 + m12 * n21
        a12 = m11 * n12 + m12 * n22
        a21 = m21 * n11 + m22 * n21
        a22 = m21 * n12 + m22 * n22

        # u_bar = dt * M^{-1} @ B @ x
        Bx0 = B0 * x
        Bx1 = B1 * x
        u0 = dt * (m11 * Bx0 + m12 * Bx1)
        u1 = dt * (m21 * Bx0 + m22 * Bx1)

        # ============================================================
        # RECURRENCE (state stays in registers)
        # ============================================================

        # h_t = A_bar @ h_{t-1} + u_bar
        new_h1 = a11 * h1 + a12 * h2 + u0
        new_h2 = a21 * h1 + a22 * h2 + u1

        # Update state (registers only)
        h1 = new_h1
        h2 = new_h2

        # Store state to HBM (fp32 -> bf16)
        tl.store(s_base + 0 * stride_s2, h1.to(tl.bfloat16), mask=d_mask)
        tl.store(s_base + 1 * stride_s2, h2.to(tl.bfloat16), mask=d_mask)


def evolution_fused_fwd(
    alpha: Tensor,
    omega: Tensor,
    dt: Tensor,
    B: Tensor,
    x: Tensor,
    initial_state: Tensor | None = None,
    eps: float = 1e-6,
    block_d: int = 128,  # Default, but autotune will override
) -> Tensor:
    """Launch fused Cayley + Evolution forward kernel.

    This eliminates the A_bar intermediate tensor entirely by computing
    the Cayley transform on-the-fly inside the scan loop.

    Args:
        alpha: Damping coefficients, shape (batch, seq, d_inner). Must be >= 0.
        omega: Frequency coefficients, shape (batch, seq, d_inner).
        dt: Timestep, shape (batch, seq, d_inner). Must be > 0.
        B: Input projection, shape (batch, seq, d_inner, 2).
        x: Input values, shape (batch, seq, d_inner).
        initial_state: Optional initial state, shape (batch, d_inner, 2).
        eps: Epsilon for numerical stability.
        block_d: Block size for d_inner dimension (autotune will override).

    Returns:
        states: Output states, shape (batch, seq, d_inner, 2).
    """
    # Ensure contiguous
    if not alpha.is_contiguous():
        alpha = alpha.contiguous()
    if not omega.is_contiguous():
        omega = omega.contiguous()
    if not dt.is_contiguous():
        dt = dt.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()

    batch, seq_len, d_inner = alpha.shape

    # Allocate output states
    states = torch.empty(
        batch, seq_len, d_inner, 2,
        dtype=torch.bfloat16,
        device=alpha.device,
    )

    # Handle initial state
    has_initial = initial_state is not None
    if has_initial:
        if not initial_state.is_contiguous():
            initial_state = initial_state.contiguous()
        init_strides = (initial_state.stride(0), initial_state.stride(1), initial_state.stride(2))
    else:
        # Dummy tensor for strides
        initial_state = alpha
        init_strides = (0, 0, 0)

    # Grid lambda for autotune - computes grid based on chosen BLOCK_D
    def grid(meta):
        return (batch, triton.cdiv(d_inner, meta["BLOCK_D"]))

    # Launch fused kernel with autotune
    evolution_fused_fwd_kernel[grid](
        alpha, omega, dt, B, x,
        initial_state,
        states,
        batch, seq_len, d_inner, has_initial,
        # alpha/omega/dt/x strides
        alpha.stride(0), alpha.stride(1), alpha.stride(2),
        # B strides
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        # initial_state strides
        init_strides[0], init_strides[1], init_strides[2],
        # states strides
        states.stride(0), states.stride(1), states.stride(2), states.stride(3),
        eps=eps,
    )

    return states


# ============================================================
# Legacy API for backward compatibility (wraps new fused kernel)
# ============================================================

def evolution_fwd(
    A_bar: Tensor,
    u_bar: Tensor,
    block_d: int = 64,
) -> Tensor:
    """Legacy evolution forward (for backward compatibility).

    WARNING: This still uses the unfused path via A_bar/u_bar.
    Use evolution_fused_fwd() for optimal performance.
    """
    # Fall back to unfused kernel for legacy callers
    return _evolution_fwd_unfused(A_bar, u_bar, block_d)


def evolution_fwd_with_initial(
    A_bar: Tensor,
    u_bar: Tensor,
    initial_state: Tensor | None = None,
    block_d: int = 64,
) -> Tensor:
    """Legacy evolution forward with initial state.

    WARNING: This still uses the unfused path via A_bar/u_bar.
    Use evolution_fused_fwd() for optimal performance.
    """
    return _evolution_fwd_with_initial_unfused(A_bar, u_bar, initial_state, block_d)


# ============================================================
# Unfused kernels (kept for backward compatibility and testing)
# ============================================================

@triton.jit
def _evolution_fwd_unfused_kernel(
    A_bar_ptr, u_bar_ptr, states_ptr,
    batch: tl.constexpr, seq_len: tl.constexpr, d_inner: tl.constexpr,
    stride_ab: tl.constexpr, stride_as: tl.constexpr, stride_ad: tl.constexpr, stride_a4: tl.constexpr,
    stride_ub: tl.constexpr, stride_us: tl.constexpr, stride_ud: tl.constexpr, stride_u2: tl.constexpr,
    stride_sb: tl.constexpr, stride_ss: tl.constexpr, stride_sd: tl.constexpr, stride_s2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Unfused evolution kernel (reads A_bar from HBM)."""
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offset = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offset < d_inner

    h1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h2 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for t in range(seq_len):
        A_base = A_bar_ptr + pid_b * stride_ab + t * stride_as + d_offset * stride_ad
        u_base = u_bar_ptr + pid_b * stride_ub + t * stride_us + d_offset * stride_ud
        s_base = states_ptr + pid_b * stride_sb + t * stride_ss + d_offset * stride_sd

        a11 = tl.load(A_base + 0 * stride_a4, mask=d_mask, other=0.0).to(tl.float32)
        a12 = tl.load(A_base + 1 * stride_a4, mask=d_mask, other=0.0).to(tl.float32)
        a21 = tl.load(A_base + 2 * stride_a4, mask=d_mask, other=0.0).to(tl.float32)
        a22 = tl.load(A_base + 3 * stride_a4, mask=d_mask, other=0.0).to(tl.float32)

        u1 = tl.load(u_base + 0 * stride_u2, mask=d_mask, other=0.0).to(tl.float32)
        u2 = tl.load(u_base + 1 * stride_u2, mask=d_mask, other=0.0).to(tl.float32)

        new_h1 = a11 * h1 + a12 * h2 + u1
        new_h2 = a21 * h1 + a22 * h2 + u2
        h1 = new_h1
        h2 = new_h2

        tl.store(s_base + 0 * stride_s2, h1.to(tl.bfloat16), mask=d_mask)
        tl.store(s_base + 1 * stride_s2, h2.to(tl.bfloat16), mask=d_mask)


def _evolution_fwd_unfused(A_bar: Tensor, u_bar: Tensor, block_d: int = 64) -> Tensor:
    """Unfused evolution forward."""
    if not A_bar.is_contiguous():
        A_bar = A_bar.contiguous()
    if not u_bar.is_contiguous():
        u_bar = u_bar.contiguous()

    batch, seq_len, d_inner, _ = A_bar.shape

    states = torch.empty(batch, seq_len, d_inner, 2, dtype=torch.bfloat16, device=A_bar.device)

    grid = (batch, triton.cdiv(d_inner, block_d))

    _evolution_fwd_unfused_kernel[grid](
        A_bar, u_bar, states,
        batch, seq_len, d_inner,
        A_bar.stride(0), A_bar.stride(1), A_bar.stride(2), A_bar.stride(3),
        u_bar.stride(0), u_bar.stride(1), u_bar.stride(2), u_bar.stride(3),
        states.stride(0), states.stride(1), states.stride(2), states.stride(3),
        BLOCK_D=block_d,
    )

    return states


@triton.jit
def _evolution_fwd_with_initial_unfused_kernel(
    A_bar_ptr, u_bar_ptr, initial_state_ptr, states_ptr,
    batch: tl.constexpr, seq_len: tl.constexpr, d_inner: tl.constexpr, has_initial: tl.constexpr,
    stride_ab: tl.constexpr, stride_as: tl.constexpr, stride_ad: tl.constexpr, stride_a4: tl.constexpr,
    stride_ub: tl.constexpr, stride_us: tl.constexpr, stride_ud: tl.constexpr, stride_u2: tl.constexpr,
    stride_ib: tl.constexpr, stride_id: tl.constexpr, stride_i2: tl.constexpr,
    stride_sb: tl.constexpr, stride_ss: tl.constexpr, stride_sd: tl.constexpr, stride_s2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Unfused evolution kernel with initial state."""
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offset = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offset < d_inner

    if has_initial:
        init_base = initial_state_ptr + pid_b * stride_ib + d_offset * stride_id
        h1 = tl.load(init_base + 0 * stride_i2, mask=d_mask, other=0.0).to(tl.float32)
        h2 = tl.load(init_base + 1 * stride_i2, mask=d_mask, other=0.0).to(tl.float32)
    else:
        h1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        h2 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for t in range(seq_len):
        A_base = A_bar_ptr + pid_b * stride_ab + t * stride_as + d_offset * stride_ad
        u_base = u_bar_ptr + pid_b * stride_ub + t * stride_us + d_offset * stride_ud
        s_base = states_ptr + pid_b * stride_sb + t * stride_ss + d_offset * stride_sd

        a11 = tl.load(A_base + 0 * stride_a4, mask=d_mask, other=0.0).to(tl.float32)
        a12 = tl.load(A_base + 1 * stride_a4, mask=d_mask, other=0.0).to(tl.float32)
        a21 = tl.load(A_base + 2 * stride_a4, mask=d_mask, other=0.0).to(tl.float32)
        a22 = tl.load(A_base + 3 * stride_a4, mask=d_mask, other=0.0).to(tl.float32)

        u1 = tl.load(u_base + 0 * stride_u2, mask=d_mask, other=0.0).to(tl.float32)
        u2 = tl.load(u_base + 1 * stride_u2, mask=d_mask, other=0.0).to(tl.float32)

        new_h1 = a11 * h1 + a12 * h2 + u1
        new_h2 = a21 * h1 + a22 * h2 + u2
        h1 = new_h1
        h2 = new_h2

        tl.store(s_base + 0 * stride_s2, h1.to(tl.bfloat16), mask=d_mask)
        tl.store(s_base + 1 * stride_s2, h2.to(tl.bfloat16), mask=d_mask)


def _evolution_fwd_with_initial_unfused(
    A_bar: Tensor, u_bar: Tensor, initial_state: Tensor | None = None, block_d: int = 64
) -> Tensor:
    """Unfused evolution forward with initial state."""
    if not A_bar.is_contiguous():
        A_bar = A_bar.contiguous()
    if not u_bar.is_contiguous():
        u_bar = u_bar.contiguous()

    batch, seq_len, d_inner, _ = A_bar.shape

    states = torch.empty(batch, seq_len, d_inner, 2, dtype=torch.bfloat16, device=A_bar.device)

    has_initial = initial_state is not None
    if has_initial:
        if not initial_state.is_contiguous():
            initial_state = initial_state.contiguous()
        init_strides = (initial_state.stride(0), initial_state.stride(1), initial_state.stride(2))
    else:
        initial_state = A_bar
        init_strides = (0, 0, 0)

    grid = (batch, triton.cdiv(d_inner, block_d))

    _evolution_fwd_with_initial_unfused_kernel[grid](
        A_bar, u_bar, initial_state, states,
        batch, seq_len, d_inner, has_initial,
        A_bar.stride(0), A_bar.stride(1), A_bar.stride(2), A_bar.stride(3),
        u_bar.stride(0), u_bar.stride(1), u_bar.stride(2), u_bar.stride(3),
        init_strides[0], init_strides[1], init_strides[2],
        states.stride(0), states.stride(1), states.stride(2), states.stride(3),
        BLOCK_D=block_d,
    )

    return states
