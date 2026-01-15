"""Autograd wrapper for KSSM evolution (scan) operation.

Strategy:
- Forward: Use FUSED Triton kernel (evolution_fused_fwd) that computes Cayley transform
           on-the-fly, eliminating the massive A_bar intermediate tensor.
- Backward: Use Triton kernel (evolution_bwd) when use_triton_backward=True,
            otherwise fall back to PyTorch reference.
"""

import torch
from torch import Tensor
from torch.autograd import Function

from kssm.kernels.evolution_fwd import (
    evolution_fwd, evolution_fwd_with_initial, evolution_fused_fwd
)
from kssm.kernels.evolution_bwd import evolution_backward_triton
from kssm.ops.reference import evolution_backward_full_reference, evolution_reference

# Global flag for backward implementation
USE_TRITON_BACKWARD = True


class EvolutionOp(Function):
    """Autograd function for KSSM state evolution.

    Forward uses Triton kernel, backward uses PyTorch reference (Phase 2).
    """

    @staticmethod
    def forward(
        ctx,
        A_bar: Tensor,
        u_bar: Tensor,
        initial_state: Tensor | None = None,
        use_triton: bool = True,
    ) -> Tensor:
        """Forward pass: compute h_t = A_bar[t] @ h_{t-1} + u_bar[t].

        Args:
            ctx: Autograd context.
            A_bar: Discretized transition matrices, shape (batch, seq, d_inner, 4).
            u_bar: Discretized inputs, shape (batch, seq, d_inner, 2).
            initial_state: Optional initial state, shape (batch, d_inner, 2).
            use_triton: Whether to use Triton kernel (True) or PyTorch reference (False).

        Returns:
            states: All states, shape (batch, seq, d_inner, 2).
        """
        # Ensure contiguous for Triton
        if not A_bar.is_contiguous():
            A_bar = A_bar.contiguous()
        if not u_bar.is_contiguous():
            u_bar = u_bar.contiguous()
        if initial_state is not None and not initial_state.is_contiguous():
            initial_state = initial_state.contiguous()

        if use_triton and A_bar.is_cuda:
            # Convert to bf16 if not already
            A_bar_bf16 = A_bar.bfloat16() if A_bar.dtype != torch.bfloat16 else A_bar
            u_bar_bf16 = u_bar.bfloat16() if u_bar.dtype != torch.bfloat16 else u_bar

            if initial_state is not None:
                init_bf16 = initial_state.bfloat16() if initial_state.dtype != torch.bfloat16 else initial_state
                states = evolution_fwd_with_initial(A_bar_bf16, u_bar_bf16, init_bf16)
            else:
                states = evolution_fwd(A_bar_bf16, u_bar_bf16)
        else:
            # Use PyTorch reference (for CPU or testing)
            states = evolution_reference(A_bar, u_bar, initial_state)
            states = states.to(A_bar.dtype)

        # Save for backward (keep original precision for gradients)
        ctx.save_for_backward(A_bar, states)
        ctx.use_triton = use_triton

        return states

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        """Backward pass using Triton kernel or PyTorch reference."""
        A_bar, states = ctx.saved_tensors

        if USE_TRITON_BACKWARD and A_bar.is_cuda:
            # Use fast Triton backward kernel
            d_A_bar, d_u_bar = evolution_backward_triton(
                A_bar, states, grad_output
            )
        else:
            # Fall back to PyTorch reference (compiled for speed)
            d_A_bar, d_u_bar = evolution_backward_full_reference(
                A_bar, states, grad_output
            )

        # Cast back to input dtype
        d_A_bar = d_A_bar.to(A_bar.dtype)
        d_u_bar = d_u_bar.to(A_bar.dtype)

        return d_A_bar, d_u_bar, None, None


def evolution(
    A_bar: Tensor,
    u_bar: Tensor,
    initial_state: Tensor | None = None,
    use_triton: bool = True,
) -> Tensor:
    """Compute KSSM state evolution with autograd support.

    This is the main entry point for the evolution operation.

    Args:
        A_bar: Discretized transition matrices, shape (batch, seq, d_inner, 4).
        u_bar: Discretized inputs, shape (batch, seq, d_inner, 2).
        initial_state: Optional initial state, shape (batch, d_inner, 2).
        use_triton: Whether to use Triton kernel (True) or PyTorch reference (False).

    Returns:
        states: All states, shape (batch, seq, d_inner, 2).

    Example:
        >>> A_bar = torch.randn(8, 256, 64, 4, device='cuda', dtype=torch.bfloat16)
        >>> u_bar = torch.randn(8, 256, 64, 2, device='cuda', dtype=torch.bfloat16)
        >>> states = evolution(A_bar, u_bar)
        >>> states.shape
        torch.Size([8, 256, 64, 2])
    """
    return EvolutionOp.apply(A_bar, u_bar, initial_state, use_triton)


def evolution_no_grad(
    A_bar: Tensor,
    u_bar: Tensor,
    initial_state: Tensor | None = None,
) -> Tensor:
    """Compute evolution without gradient tracking (inference mode).

    Directly calls Triton kernel without autograd overhead.

    Args:
        A_bar: Discretized transition matrices, shape (batch, seq, d_inner, 4).
        u_bar: Discretized inputs, shape (batch, seq, d_inner, 2).
        initial_state: Optional initial state, shape (batch, d_inner, 2).

    Returns:
        states: All states, shape (batch, seq, d_inner, 2).
    """
    if not A_bar.is_contiguous():
        A_bar = A_bar.contiguous()
    if not u_bar.is_contiguous():
        u_bar = u_bar.contiguous()

    A_bar_bf16 = A_bar.bfloat16() if A_bar.dtype != torch.bfloat16 else A_bar
    u_bar_bf16 = u_bar.bfloat16() if u_bar.dtype != torch.bfloat16 else u_bar

    if initial_state is not None:
        if not initial_state.is_contiguous():
            initial_state = initial_state.contiguous()
        init_bf16 = initial_state.bfloat16() if initial_state.dtype != torch.bfloat16 else initial_state
        return evolution_fwd_with_initial(A_bar_bf16, u_bar_bf16, init_bf16)
    else:
        return evolution_fwd(A_bar_bf16, u_bar_bf16)


# ============================================================
# FUSED Cayley + Evolution (eliminates A_bar intermediate tensor)
# ============================================================

class FusedEvolutionOp(Function):
    """Autograd function for FUSED Cayley transform + Evolution.

    This version computes the Cayley discretization ON-THE-FLY inside the
    scan loop, eliminating the massive A_bar intermediate tensor from HBM.

    Memory savings at L=8192, d_inner=1536, batch=4:
    - Before: A_bar = 4 × 8192 × 1536 × 4 × 2 bytes = ~400 MB
    - After: 0 bytes (computed in registers)
    """

    @staticmethod
    def forward(
        ctx,
        alpha: Tensor,
        omega: Tensor,
        dt: Tensor,
        B: Tensor,
        x: Tensor,
        initial_state: Tensor | None = None,
        eps: float = 1e-6,
        use_triton: bool = True,
    ) -> Tensor:
        """Fused forward pass: Cayley transform + evolution in one kernel.

        Args:
            ctx: Autograd context.
            alpha: Damping coefficients (post-softplus), shape (batch, seq, d_inner).
            omega: Frequency coefficients, shape (batch, seq, d_inner).
            dt: Timestep (post-softplus), shape (batch, seq, d_inner).
            B: Input projection, shape (batch, seq, d_inner, 2).
            x: Input values, shape (batch, seq, d_inner).
            initial_state: Optional initial state, shape (batch, d_inner, 2).
            eps: Numerical stability epsilon.
            use_triton: Whether to use Triton kernel.

        Returns:
            states: All states, shape (batch, seq, d_inner, 2).
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
        if initial_state is not None and not initial_state.is_contiguous():
            initial_state = initial_state.contiguous()

        if use_triton and alpha.is_cuda:
            # Convert to bf16 for Triton kernel
            alpha_bf16 = alpha.bfloat16() if alpha.dtype != torch.bfloat16 else alpha
            omega_bf16 = omega.bfloat16() if omega.dtype != torch.bfloat16 else omega
            dt_bf16 = dt.bfloat16() if dt.dtype != torch.bfloat16 else dt
            B_bf16 = B.bfloat16() if B.dtype != torch.bfloat16 else B
            x_bf16 = x.bfloat16() if x.dtype != torch.bfloat16 else x

            init_bf16 = None
            if initial_state is not None:
                init_bf16 = initial_state.bfloat16() if initial_state.dtype != torch.bfloat16 else initial_state

            # Call FUSED kernel - no A_bar materialized!
            states = evolution_fused_fwd(
                alpha_bf16, omega_bf16, dt_bf16, B_bf16, x_bf16,
                initial_state=init_bf16,
                eps=eps,
            )
        else:
            # Fall back to unfused path for CPU/testing
            from kssm.kernels.cayley_fused import cayley_fused_pytorch
            A_bar, u_bar = cayley_fused_pytorch(alpha, omega, dt, B, x, eps)
            states = evolution_reference(A_bar, u_bar, initial_state)
            states = states.to(alpha.dtype)

        # Save for backward - we need to recompute A_bar in backward
        # Save the raw params instead of the massive A_bar tensor
        ctx.save_for_backward(alpha, omega, dt, B, x, states)
        ctx.eps = eps
        ctx.use_triton = use_triton

        return states

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        """Backward pass for fused evolution.

        Note: We need to recompute A_bar for backward. This is still more
        efficient than the unfused path because we only allocate A_bar once
        (during backward) instead of twice (forward + backward).
        """
        alpha, omega, dt, B, x, states = ctx.saved_tensors

        # Recompute A_bar and u_bar for backward (cheaper than storing)
        from kssm.kernels.cayley_fused import cayley_fused_pytorch

        # Use PyTorch version with autograd for backward
        with torch.enable_grad():
            alpha_g = alpha.detach().float().requires_grad_(True)
            omega_g = omega.detach().float().requires_grad_(True)
            dt_g = dt.detach().float().requires_grad_(True)
            B_g = B.detach().float().requires_grad_(True)
            x_g = x.detach().float().requires_grad_(True)

            # Compute A_bar, u_bar with autograd tracking
            A_bar_g, u_bar_g = cayley_fused_pytorch(alpha_g, omega_g, dt_g, B_g, x_g, ctx.eps)

            # Get gradients from evolution backward
            if USE_TRITON_BACKWARD and alpha.is_cuda:
                d_A_bar, d_u_bar = evolution_backward_triton(
                    A_bar_g.to(states.dtype).detach(), states, grad_output
                )
            else:
                d_A_bar, d_u_bar = evolution_backward_full_reference(
                    A_bar_g.detach(), states.float(), grad_output.float()
                )

            # Backward through Cayley transform
            torch.autograd.backward(
                [A_bar_g, u_bar_g],
                [d_A_bar.float(), d_u_bar.float()],
            )

        d_alpha = alpha_g.grad.to(alpha.dtype) if alpha_g.grad is not None else None
        d_omega = omega_g.grad.to(omega.dtype) if omega_g.grad is not None else None
        d_dt = dt_g.grad.to(dt.dtype) if dt_g.grad is not None else None
        d_B = B_g.grad.to(B.dtype) if B_g.grad is not None else None
        d_x = x_g.grad.to(x.dtype) if x_g.grad is not None else None

        return d_alpha, d_omega, d_dt, d_B, d_x, None, None, None


def evolution_fused(
    alpha: Tensor,
    omega: Tensor,
    dt: Tensor,
    B: Tensor,
    x: Tensor,
    initial_state: Tensor | None = None,
    eps: float = 1e-6,
    use_triton: bool = True,
) -> Tensor:
    """Fused Cayley transform + evolution with autograd support.

    This is the OPTIMIZED entry point that eliminates the A_bar intermediate
    tensor by computing the Cayley discretization on-the-fly in the scan kernel.

    Memory savings at L=8192, d_inner=1536, batch=4: ~400 MB

    Args:
        alpha: Damping coefficients (post-softplus), shape (batch, seq, d_inner).
        omega: Frequency coefficients, shape (batch, seq, d_inner).
        dt: Timestep (post-softplus), shape (batch, seq, d_inner).
        B: Input projection, shape (batch, seq, d_inner, 2).
        x: Input values, shape (batch, seq, d_inner).
        initial_state: Optional initial state, shape (batch, d_inner, 2).
        eps: Numerical stability epsilon.
        use_triton: Whether to use Triton kernel.

    Returns:
        states: All states, shape (batch, seq, d_inner, 2).
    """
    return FusedEvolutionOp.apply(alpha, omega, dt, B, x, initial_state, eps, use_triton)


def evolution_fused_no_grad(
    alpha: Tensor,
    omega: Tensor,
    dt: Tensor,
    B: Tensor,
    x: Tensor,
    initial_state: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Fused evolution without gradient tracking (inference mode).

    Args:
        alpha: Damping coefficients, shape (batch, seq, d_inner).
        omega: Frequency coefficients, shape (batch, seq, d_inner).
        dt: Timestep, shape (batch, seq, d_inner).
        B: Input projection, shape (batch, seq, d_inner, 2).
        x: Input values, shape (batch, seq, d_inner).
        initial_state: Optional initial state, shape (batch, d_inner, 2).
        eps: Numerical stability epsilon.

    Returns:
        states: All states, shape (batch, seq, d_inner, 2).
    """
    # Ensure contiguous and bf16
    alpha_bf16 = alpha.bfloat16().contiguous() if alpha.dtype != torch.bfloat16 else alpha.contiguous()
    omega_bf16 = omega.bfloat16().contiguous() if omega.dtype != torch.bfloat16 else omega.contiguous()
    dt_bf16 = dt.bfloat16().contiguous() if dt.dtype != torch.bfloat16 else dt.contiguous()
    B_bf16 = B.bfloat16().contiguous() if B.dtype != torch.bfloat16 else B.contiguous()
    x_bf16 = x.bfloat16().contiguous() if x.dtype != torch.bfloat16 else x.contiguous()

    init_bf16 = None
    if initial_state is not None:
        init_bf16 = initial_state.bfloat16().contiguous() if initial_state.dtype != torch.bfloat16 else initial_state.contiguous()

    return evolution_fused_fwd(
        alpha_bf16, omega_bf16, dt_bf16, B_bf16, x_bf16,
        initial_state=init_bf16,
        eps=eps,
    )
