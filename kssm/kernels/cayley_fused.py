"""Fused Triton kernel for Cayley transform discretization.

Computes the discretized transition matrix A_bar and input u_bar in a single
fused kernel, avoiding intermediate HBM writes.

Mathematical background:
- Continuous dynamics: dz/dt = A(t)z + Bu where A = [[-α, ω], [-ω, -α]]
- Cayley transform: A_bar = (I - τA)^{-1}(I + τA), τ = dt/2
- For 2x2 blocks, the inverse has analytical form with det(M) = (1+τα)² + (τω)²

Key optimizations:
- Fused computation avoids writing intermediate M, M^{-1} to HBM
- Epsilon added to determinant for numerical stability (computed in fp32)
- Output in bf16 for memory efficiency
"""

import triton
import triton.language as tl
import torch
from torch import Tensor


@triton.jit
def cayley_fused_kernel(
    # Inputs
    alpha_ptr,  # (batch, seq, d_inner) - damping, must be >= 0
    omega_ptr,  # (batch, seq, d_inner) - frequency
    dt_ptr,     # (batch, seq, d_inner) - timestep
    B_ptr,      # (batch, seq, d_inner, 2) - input projection
    x_ptr,      # (batch, seq, d_inner) - input values
    # Outputs
    A_bar_ptr,  # (batch, seq, d_inner, 4) - discretized transition
    u_bar_ptr,  # (batch, seq, d_inner, 2) - discretized input
    # Dimensions
    seq_len,
    d_inner,
    # Strides for alpha/omega/dt/x (batch, seq, d_inner)
    stride_pb,  # param batch stride
    stride_ps,  # param seq stride
    stride_pd,  # param d_inner stride
    # Strides for B (batch, seq, d_inner, 2)
    stride_Bb,
    stride_Bs,
    stride_Bd,
    stride_B2,
    # Strides for A_bar (batch, seq, d_inner, 4)
    stride_Ab,
    stride_As,
    stride_Ad,
    stride_A4,
    # Strides for u_bar (batch, seq, d_inner, 2)
    stride_ub,
    stride_us,
    stride_ud,
    stride_u2,
    # Epsilon for numerical stability
    eps,
    # Block size
    BLOCK_D: tl.constexpr,
):
    """Fused Cayley transform kernel.

    Grid: (batch, seq_len, cdiv(d_inner, BLOCK_D))
    Each program handles one (batch, seq) position and BLOCK_D features.
    """
    pid_b = tl.program_id(0)  # Batch index
    pid_s = tl.program_id(1)  # Sequence index
    pid_d = tl.program_id(2)  # Feature block index

    # Feature offsets
    d_offset = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offset < d_inner

    # Compute base offset for this (batch, seq) position
    base_offset = pid_b * stride_pb + pid_s * stride_ps

    # Load parameters (bf16 -> fp32 for precision)
    alpha = tl.load(alpha_ptr + base_offset + d_offset * stride_pd, mask=d_mask, other=0.0).to(tl.float32)
    omega = tl.load(omega_ptr + base_offset + d_offset * stride_pd, mask=d_mask, other=0.0).to(tl.float32)
    dt = tl.load(dt_ptr + base_offset + d_offset * stride_pd, mask=d_mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + base_offset + d_offset * stride_pd, mask=d_mask, other=0.0).to(tl.float32)

    # Load B (2 components)
    B_base = pid_b * stride_Bb + pid_s * stride_Bs
    B0 = tl.load(B_ptr + B_base + d_offset * stride_Bd + 0 * stride_B2, mask=d_mask, other=0.0).to(tl.float32)
    B1 = tl.load(B_ptr + B_base + d_offset * stride_Bd + 1 * stride_B2, mask=d_mask, other=0.0).to(tl.float32)

    # Cayley transform computation (all in fp32)
    tau = dt * 0.5

    # Determinant of M = I - τA where A = [[-α, ω], [-ω, -α]]
    # M = [[1 + τα, -τω], [τω, 1 + τα]]
    # det(M) = (1 + τα)² + (τω)²
    one_plus_tau_alpha = 1.0 + tau * alpha
    tau_omega = tau * omega
    det_M = one_plus_tau_alpha * one_plus_tau_alpha + tau_omega * tau_omega

    # Numerical stability: add epsilon (critical per plan checklist)
    inv_det = 1.0 / (det_M + eps)

    # M^{-1} components
    # M^{-1} = (1/det) * [[1 + τα, τω], [-τω, 1 + τα]]
    m11 = one_plus_tau_alpha * inv_det
    m12 = tau_omega * inv_det
    m21 = -tau_omega * inv_det
    m22 = one_plus_tau_alpha * inv_det

    # N = I + τA where A = [[-α, ω], [-ω, -α]]
    # N = [[1 - τα, τω], [-τω, 1 - τα]]
    one_minus_tau_alpha = 1.0 - tau * alpha
    n11 = one_minus_tau_alpha
    n12 = tau_omega
    n21 = -tau_omega
    n22 = one_minus_tau_alpha

    # A_bar = M^{-1} @ N (2x2 matrix multiply)
    a11 = m11 * n11 + m12 * n21
    a12 = m11 * n12 + m12 * n22
    a21 = m21 * n11 + m22 * n21
    a22 = m21 * n12 + m22 * n22

    # u_bar = dt * M^{-1} @ B @ x
    # B @ x = [B0 * x, B1 * x]
    Bx0 = B0 * x
    Bx1 = B1 * x

    # M^{-1} @ (B @ x)
    u0 = m11 * Bx0 + m12 * Bx1
    u1 = m21 * Bx0 + m22 * Bx1

    # Scale by dt
    u0 = dt * u0
    u1 = dt * u1

    # Store A_bar (fp32 -> bf16)
    A_base = pid_b * stride_Ab + pid_s * stride_As
    tl.store(A_bar_ptr + A_base + d_offset * stride_Ad + 0 * stride_A4, a11.to(tl.bfloat16), mask=d_mask)
    tl.store(A_bar_ptr + A_base + d_offset * stride_Ad + 1 * stride_A4, a12.to(tl.bfloat16), mask=d_mask)
    tl.store(A_bar_ptr + A_base + d_offset * stride_Ad + 2 * stride_A4, a21.to(tl.bfloat16), mask=d_mask)
    tl.store(A_bar_ptr + A_base + d_offset * stride_Ad + 3 * stride_A4, a22.to(tl.bfloat16), mask=d_mask)

    # Store u_bar (fp32 -> bf16)
    u_base = pid_b * stride_ub + pid_s * stride_us
    tl.store(u_bar_ptr + u_base + d_offset * stride_ud + 0 * stride_u2, u0.to(tl.bfloat16), mask=d_mask)
    tl.store(u_bar_ptr + u_base + d_offset * stride_ud + 1 * stride_u2, u1.to(tl.bfloat16), mask=d_mask)


def cayley_fused(
    alpha: Tensor,
    omega: Tensor,
    dt: Tensor,
    B: Tensor,
    x: Tensor,
    eps: float = 1e-6,
    block_d: int = 64,
) -> tuple[Tensor, Tensor]:
    """Launch fused Cayley transform kernel.

    Args:
        alpha: Damping coefficients, shape (batch, seq, d_inner). Must be >= 0.
        omega: Frequency coefficients, shape (batch, seq, d_inner).
        dt: Timestep, shape (batch, seq, d_inner). Must be > 0.
        B: Input projection, shape (batch, seq, d_inner, 2).
        x: Input values, shape (batch, seq, d_inner).
        eps: Epsilon for numerical stability in determinant.
        block_d: Block size for d_inner dimension.

    Returns:
        A_bar: Discretized transition matrix, shape (batch, seq, d_inner, 4).
        u_bar: Discretized input, shape (batch, seq, d_inner, 2).
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

    # Allocate outputs
    A_bar = torch.empty(
        batch, seq_len, d_inner, 4,
        dtype=torch.bfloat16,
        device=alpha.device,
    )
    u_bar = torch.empty(
        batch, seq_len, d_inner, 2,
        dtype=torch.bfloat16,
        device=alpha.device,
    )

    # Compute grid
    grid = (batch, seq_len, triton.cdiv(d_inner, block_d))

    # Launch kernel
    cayley_fused_kernel[grid](
        alpha, omega, dt, B, x,
        A_bar, u_bar,
        seq_len, d_inner,
        # alpha/omega/dt/x strides
        alpha.stride(0), alpha.stride(1), alpha.stride(2),
        # B strides
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        # A_bar strides
        A_bar.stride(0), A_bar.stride(1), A_bar.stride(2), A_bar.stride(3),
        # u_bar strides
        u_bar.stride(0), u_bar.stride(1), u_bar.stride(2), u_bar.stride(3),
        eps=eps,
        BLOCK_D=block_d,
    )

    return A_bar, u_bar


def cayley_fused_pytorch(
    alpha: Tensor,
    omega: Tensor,
    dt: Tensor,
    B: Tensor,
    x: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """PyTorch reference implementation of fused Cayley transform.

    For testing and CPU fallback.
    """
    tau = dt / 2.0

    # Determinant
    one_plus_tau_alpha = 1.0 + tau * alpha
    tau_omega = tau * omega
    det_M = one_plus_tau_alpha ** 2 + tau_omega ** 2
    inv_det = 1.0 / (det_M + eps)

    # M^{-1} components
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

    A_bar = torch.stack([a11, a12, a21, a22], dim=-1)

    # u_bar = dt * M^{-1} @ B @ x
    Bx0 = B[..., 0] * x
    Bx1 = B[..., 1] * x

    u0 = dt * (m11 * Bx0 + m12 * Bx1)
    u1 = dt * (m21 * Bx0 + m22 * Bx1)

    u_bar = torch.stack([u0, u1], dim=-1)

    return A_bar, u_bar
