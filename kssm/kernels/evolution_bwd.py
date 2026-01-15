"""Triton kernel for backward evolution (reverse adjoint scan).

This implements the adjoint state method for computing gradients:
    λ_{t-1} = A_t^T @ λ_t + ∂L/∂h_{t-1}

The adjoint equation runs backwards through time, accumulating gradients.

Key insight: A^T for our 2x2 block just swaps the off-diagonal elements:
    A = [[a11, a12], [a21, a22]]
    A^T = [[a11, a21], [a12, a22]]

V1 Strategy (Modular):
- Kernel computes: λ (adjoint states) and d_u_bar (gradient w.r.t. input)
- PyTorch computes: d_A_bar using λ and saved forward states (outer product)
"""

import triton
import triton.language as tl
import torch
from torch import Tensor


@triton.jit
def evolution_bwd_kernel(
    # Inputs
    A_bar_ptr,      # (batch, seq, d_inner, 4) - discretized transition matrices
    grad_out_ptr,   # (batch, seq, d_inner, 2) - gradient w.r.t. output states
    # Outputs
    d_u_bar_ptr,    # (batch, seq, d_inner, 2) - gradient w.r.t. input
    # Dimensions
    batch: tl.constexpr,
    seq_len: tl.constexpr,
    d_inner: tl.constexpr,
    # Strides for A_bar (batch, seq, d_inner, 4)
    stride_ab: tl.constexpr,
    stride_as: tl.constexpr,
    stride_ad: tl.constexpr,
    stride_a4: tl.constexpr,
    # Strides for grad_out (batch, seq, d_inner, 2)
    stride_gb: tl.constexpr,
    stride_gs: tl.constexpr,
    stride_gd: tl.constexpr,
    stride_g2: tl.constexpr,
    # Strides for d_u_bar (batch, seq, d_inner, 2)
    stride_db: tl.constexpr,
    stride_ds: tl.constexpr,
    stride_dd: tl.constexpr,
    stride_d2: tl.constexpr,
    # Block size
    BLOCK_D: tl.constexpr,
):
    """Reverse adjoint scan kernel.

    Computes:
        λ_{t-1} = A_t^T @ λ_t + grad_out_{t-1}
        d_u_bar[t] = λ_t

    Grid: (batch, cdiv(d_inner, BLOCK_D))
    """
    # Program IDs
    pid_b = tl.program_id(0)  # Batch index
    pid_d = tl.program_id(1)  # Feature block index

    # Compute feature dimension offsets
    d_offset = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offset < d_inner

    # Initialize adjoint state (λ) to zeros
    # λ_T = 0 (terminal condition for adjoint)
    lambda1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    lambda2 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Reverse loop through sequence
    for t_idx in range(seq_len):
        # Compute actual time index (reverse order)
        t = seq_len - 1 - t_idx

        # Load grad_out at time t (∂L/∂h_t)
        g_base = grad_out_ptr + pid_b * stride_gb + t * stride_gs + d_offset * stride_gd
        grad1 = tl.load(g_base + 0 * stride_g2, mask=d_mask, other=0.0).to(tl.float32)
        grad2 = tl.load(g_base + 1 * stride_g2, mask=d_mask, other=0.0).to(tl.float32)

        # Add direct gradient contribution to λ
        # λ_t = λ_t + ∂L/∂h_t
        lambda1 = lambda1 + grad1
        lambda2 = lambda2 + grad2

        # Store d_u_bar[t] = λ_t (gradient w.r.t. input at time t)
        d_base = d_u_bar_ptr + pid_b * stride_db + t * stride_ds + d_offset * stride_dd
        tl.store(d_base + 0 * stride_d2, lambda1.to(tl.bfloat16), mask=d_mask)
        tl.store(d_base + 1 * stride_d2, lambda2.to(tl.bfloat16), mask=d_mask)

        # Propagate adjoint: λ_{t-1} = A_t^T @ λ_t
        # Only propagate if not at t=0 (we don't need λ_{-1})
        if t > 0:
            # Load A_bar at time t
            A_base = A_bar_ptr + pid_b * stride_ab + t * stride_as + d_offset * stride_ad
            a11 = tl.load(A_base + 0 * stride_a4, mask=d_mask, other=0.0).to(tl.float32)
            a12 = tl.load(A_base + 1 * stride_a4, mask=d_mask, other=0.0).to(tl.float32)
            a21 = tl.load(A_base + 2 * stride_a4, mask=d_mask, other=0.0).to(tl.float32)
            a22 = tl.load(A_base + 3 * stride_a4, mask=d_mask, other=0.0).to(tl.float32)

            # Apply A^T: swap a12 and a21
            # A^T @ λ = [[a11, a21], [a12, a22]] @ [λ1, λ2]
            new_lambda1 = a11 * lambda1 + a21 * lambda2
            new_lambda2 = a12 * lambda1 + a22 * lambda2

            lambda1 = new_lambda1
            lambda2 = new_lambda2


def evolution_bwd(
    A_bar: Tensor,
    grad_output: Tensor,
    block_d: int = 64,
) -> Tensor:
    """Launch evolution backward kernel.

    Computes gradients w.r.t. u_bar using the adjoint state method.

    Args:
        A_bar: Discretized transition matrices, shape (batch, seq, d_inner, 4).
               Must be contiguous and bfloat16.
        grad_output: Gradient w.r.t. output states, shape (batch, seq, d_inner, 2).
                     Must be contiguous and bfloat16.
        block_d: Block size for d_inner dimension.

    Returns:
        d_u_bar: Gradient w.r.t. input, shape (batch, seq, d_inner, 2), bfloat16.
    """
    # Ensure contiguous
    if not A_bar.is_contiguous():
        A_bar = A_bar.contiguous()
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()

    batch, seq_len, d_inner, _ = A_bar.shape

    # Allocate output
    d_u_bar = torch.empty(
        batch, seq_len, d_inner, 2,
        dtype=torch.bfloat16,
        device=A_bar.device,
    )

    # Compute grid
    grid = (batch, triton.cdiv(d_inner, block_d))

    # Launch kernel
    evolution_bwd_kernel[grid](
        A_bar, grad_output, d_u_bar,
        batch, seq_len, d_inner,
        # A_bar strides
        A_bar.stride(0), A_bar.stride(1), A_bar.stride(2), A_bar.stride(3),
        # grad_output strides
        grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
        # d_u_bar strides
        d_u_bar.stride(0), d_u_bar.stride(1), d_u_bar.stride(2), d_u_bar.stride(3),
        # Block size
        BLOCK_D=block_d,
    )

    return d_u_bar


@triton.jit
def compute_d_A_bar_kernel(
    # Inputs
    adjoint_states_ptr,  # (batch, seq, d_inner, 2) - λ values
    forward_states_ptr,  # (batch, seq, d_inner, 2) - h values
    # Output
    d_A_bar_ptr,         # (batch, seq, d_inner, 4) - gradient output
    # Dimensions
    batch: tl.constexpr,
    seq_len: tl.constexpr,
    d_inner: tl.constexpr,
    # Strides for adjoint_states (batch, seq, d_inner, 2)
    stride_lb: tl.constexpr,
    stride_ls: tl.constexpr,
    stride_ld: tl.constexpr,
    stride_l2: tl.constexpr,
    # Strides for forward_states (batch, seq, d_inner, 2)
    stride_hb: tl.constexpr,
    stride_hs: tl.constexpr,
    stride_hd: tl.constexpr,
    stride_h2: tl.constexpr,
    # Strides for d_A_bar (batch, seq, d_inner, 4)
    stride_ab: tl.constexpr,
    stride_as: tl.constexpr,
    stride_ad: tl.constexpr,
    stride_a4: tl.constexpr,
    # Block sizes
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute d_A_bar = outer(λ_t, h_{t-1}) for all timesteps.

    Grid: (batch, cdiv(seq_len, BLOCK_S), cdiv(d_inner, BLOCK_D))
    """
    pid_b = tl.program_id(0)  # Batch index
    pid_s = tl.program_id(1)  # Sequence block index
    pid_d = tl.program_id(2)  # Feature block index

    # Compute offsets
    s_offset = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offset = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # Create masks
    s_mask = s_offset < seq_len
    d_mask = d_offset < d_inner

    # Combined mask for 2D block
    # Shape: (BLOCK_S, BLOCK_D)
    mask = s_mask[:, None] & d_mask[None, :]

    # Load λ_t (adjoint states at time t)
    # Base pointer for this batch, sequence block, feature block
    lambda_base = adjoint_states_ptr + pid_b * stride_lb
    lambda1_ptrs = lambda_base + s_offset[:, None] * stride_ls + d_offset[None, :] * stride_ld + 0 * stride_l2
    lambda2_ptrs = lambda_base + s_offset[:, None] * stride_ls + d_offset[None, :] * stride_ld + 1 * stride_l2

    lambda1 = tl.load(lambda1_ptrs, mask=mask, other=0.0)
    lambda2 = tl.load(lambda2_ptrs, mask=mask, other=0.0)

    # Load h_{t-1} (forward states at time t-1)
    # For t=0, h_{-1} = 0, so we use s_offset - 1 with masking
    s_prev = s_offset - 1
    s_prev_mask = (s_prev >= 0) & (s_prev < seq_len)
    prev_mask = s_prev_mask[:, None] & d_mask[None, :]

    h_base = forward_states_ptr + pid_b * stride_hb
    h1_prev_ptrs = h_base + s_prev[:, None] * stride_hs + d_offset[None, :] * stride_hd + 0 * stride_h2
    h2_prev_ptrs = h_base + s_prev[:, None] * stride_hs + d_offset[None, :] * stride_hd + 1 * stride_h2

    h1_prev = tl.load(h1_prev_ptrs, mask=prev_mask, other=0.0)
    h2_prev = tl.load(h2_prev_ptrs, mask=prev_mask, other=0.0)

    # Compute outer product: d_A_bar[t] = outer(λ_t, h_{t-1})
    da11 = lambda1 * h1_prev
    da12 = lambda1 * h2_prev
    da21 = lambda2 * h1_prev
    da22 = lambda2 * h2_prev

    # Store d_A_bar
    d_A_base = d_A_bar_ptr + pid_b * stride_ab
    da11_ptrs = d_A_base + s_offset[:, None] * stride_as + d_offset[None, :] * stride_ad + 0 * stride_a4
    da12_ptrs = d_A_base + s_offset[:, None] * stride_as + d_offset[None, :] * stride_ad + 1 * stride_a4
    da21_ptrs = d_A_base + s_offset[:, None] * stride_as + d_offset[None, :] * stride_ad + 2 * stride_a4
    da22_ptrs = d_A_base + s_offset[:, None] * stride_as + d_offset[None, :] * stride_ad + 3 * stride_a4

    tl.store(da11_ptrs, da11, mask=mask)
    tl.store(da12_ptrs, da12, mask=mask)
    tl.store(da21_ptrs, da21, mask=mask)
    tl.store(da22_ptrs, da22, mask=mask)


def compute_d_A_bar_triton(
    adjoint_states: Tensor,
    forward_states: Tensor,
    block_s: int = 32,
    block_d: int = 32,
) -> Tensor:
    """Compute d_A_bar using Triton kernel.

    Args:
        adjoint_states: Adjoint states λ, shape (batch, seq, d_inner, 2).
        forward_states: Forward states h, shape (batch, seq, d_inner, 2).
        block_s: Block size for sequence dimension.
        block_d: Block size for feature dimension.

    Returns:
        d_A_bar: Gradient w.r.t. A_bar, shape (batch, seq, d_inner, 4).
    """
    batch, seq_len, d_inner, _ = adjoint_states.shape
    dtype = adjoint_states.dtype

    # Ensure contiguous
    if not adjoint_states.is_contiguous():
        adjoint_states = adjoint_states.contiguous()
    if not forward_states.is_contiguous():
        forward_states = forward_states.contiguous()

    # Allocate output
    d_A_bar = torch.empty(
        batch, seq_len, d_inner, 4,
        dtype=dtype,
        device=adjoint_states.device,
    )

    # Compute grid
    grid = (batch, triton.cdiv(seq_len, block_s), triton.cdiv(d_inner, block_d))

    # Launch kernel
    compute_d_A_bar_kernel[grid](
        adjoint_states, forward_states, d_A_bar,
        batch, seq_len, d_inner,
        # adjoint_states strides
        adjoint_states.stride(0), adjoint_states.stride(1),
        adjoint_states.stride(2), adjoint_states.stride(3),
        # forward_states strides
        forward_states.stride(0), forward_states.stride(1),
        forward_states.stride(2), forward_states.stride(3),
        # d_A_bar strides
        d_A_bar.stride(0), d_A_bar.stride(1),
        d_A_bar.stride(2), d_A_bar.stride(3),
        # Block sizes
        BLOCK_S=block_s,
        BLOCK_D=block_d,
    )

    return d_A_bar


def compute_d_A_bar(
    adjoint_states: Tensor,
    forward_states: Tensor,
) -> Tensor:
    """Compute gradient w.r.t. A_bar using adjoint states and forward states.

    This is the "outer product" part of the backward pass:
        d_A_bar[t] = outer(λ_t, h_{t-1})

    For 2x2 blocks:
        d_A_bar[t, :, 0] = λ_1 * h1_{t-1}  (da11)
        d_A_bar[t, :, 1] = λ_1 * h2_{t-1}  (da12)
        d_A_bar[t, :, 2] = λ_2 * h1_{t-1}  (da21)
        d_A_bar[t, :, 3] = λ_2 * h2_{t-1}  (da22)

    Uses Triton kernel on CUDA, falls back to PyTorch on CPU.

    Args:
        adjoint_states: Adjoint states λ, shape (batch, seq, d_inner, 2).
                        This is the same as d_u_bar from evolution_bwd.
        forward_states: Forward states h, shape (batch, seq, d_inner, 2).

    Returns:
        d_A_bar: Gradient w.r.t. A_bar, shape (batch, seq, d_inner, 4).
    """
    # Use Triton kernel on CUDA
    if adjoint_states.is_cuda:
        return compute_d_A_bar_triton(adjoint_states, forward_states)

    # PyTorch fallback for CPU
    return _compute_d_A_bar_pytorch(adjoint_states, forward_states)


def _compute_d_A_bar_pytorch(
    adjoint_states: Tensor,
    forward_states: Tensor,
) -> Tensor:
    """PyTorch reference implementation for compute_d_A_bar."""
    batch, seq_len, d_inner, _ = adjoint_states.shape
    device = adjoint_states.device
    dtype = adjoint_states.dtype

    # Prepend zero for h_{-1} (initial state was zero)
    # h_prev[t] = h_{t-1}
    h_prev = torch.cat([
        torch.zeros(batch, 1, d_inner, 2, dtype=dtype, device=device),
        forward_states[:, :-1, :, :]
    ], dim=1)

    # Extract components
    lambda1 = adjoint_states[..., 0]  # (batch, seq, d_inner)
    lambda2 = adjoint_states[..., 1]
    h1_prev = h_prev[..., 0]
    h2_prev = h_prev[..., 1]

    # Compute outer products
    da11 = lambda1 * h1_prev
    da12 = lambda1 * h2_prev
    da21 = lambda2 * h1_prev
    da22 = lambda2 * h2_prev

    # Stack into (batch, seq, d_inner, 4)
    d_A_bar = torch.stack([da11, da12, da21, da22], dim=-1)

    return d_A_bar


def evolution_backward_triton(
    A_bar: Tensor,
    forward_states: Tensor,
    grad_output: Tensor,
) -> tuple[Tensor, Tensor]:
    """Complete backward pass using Triton kernel + PyTorch.

    V1 Strategy:
    - Triton kernel computes: d_u_bar (= λ, adjoint states)
    - PyTorch computes: d_A_bar using outer product

    Args:
        A_bar: Discretized transition matrices, shape (batch, seq, d_inner, 4).
        forward_states: Forward states h, shape (batch, seq, d_inner, 2).
        grad_output: Gradient w.r.t. output states, shape (batch, seq, d_inner, 2).

    Returns:
        d_A_bar: Gradient w.r.t. A_bar, shape (batch, seq, d_inner, 4).
        d_u_bar: Gradient w.r.t. u_bar, shape (batch, seq, d_inner, 2).
    """
    # Save original dtype for output
    original_dtype = A_bar.dtype

    # Convert to bf16 for Triton kernel (required by kernel)
    A_bar_bf16 = A_bar.bfloat16() if A_bar.dtype != torch.bfloat16 else A_bar
    grad_output_bf16 = grad_output.bfloat16() if grad_output.dtype != torch.bfloat16 else grad_output

    # Triton: compute d_u_bar (adjoint states)
    d_u_bar = evolution_bwd(A_bar_bf16, grad_output_bf16)

    # Convert d_u_bar to float32 for more accurate d_A_bar computation
    d_u_bar_f32 = d_u_bar.float()
    forward_states_f32 = forward_states.float()

    # PyTorch: compute d_A_bar using outer product in float32
    d_A_bar = compute_d_A_bar(d_u_bar_f32, forward_states_f32)

    # Cast back to bf16 for consistency (computation was in float32)
    d_A_bar = d_A_bar.bfloat16()

    return d_A_bar, d_u_bar
