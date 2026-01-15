"""Autograd wrapper for Cayley transform discretization.

Provides autograd-compatible Cayley transform that can be used in training.
Uses Triton kernel for forward, PyTorch for backward (Phase 2 strategy).
"""

import torch
from torch import Tensor
from torch.autograd import Function

from kssm.kernels.cayley_fused import cayley_fused, cayley_fused_pytorch


class CayleyTransformOp(Function):
    """Autograd function for Cayley transform discretization.

    Forward uses Triton kernel, backward uses PyTorch (Phase 2).
    """

    @staticmethod
    def forward(
        ctx,
        alpha: Tensor,
        omega: Tensor,
        dt: Tensor,
        B: Tensor,
        x: Tensor,
        eps: float = 1e-6,
        use_triton: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Compute Cayley transform discretization.

        Args:
            ctx: Autograd context.
            alpha: Damping coefficients (batch, seq, d_inner). Must be >= 0.
            omega: Frequency coefficients (batch, seq, d_inner).
            dt: Timestep (batch, seq, d_inner). Must be > 0.
            B: Input projection (batch, seq, d_inner, 2).
            x: Input values (batch, seq, d_inner).
            eps: Numerical stability epsilon.
            use_triton: Whether to use Triton kernel.

        Returns:
            A_bar: Discretized transition (batch, seq, d_inner, 4).
            u_bar: Discretized input (batch, seq, d_inner, 2).
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

        if use_triton and alpha.is_cuda:
            # Convert to bf16 for Triton
            alpha_bf16 = alpha.bfloat16() if alpha.dtype != torch.bfloat16 else alpha
            omega_bf16 = omega.bfloat16() if omega.dtype != torch.bfloat16 else omega
            dt_bf16 = dt.bfloat16() if dt.dtype != torch.bfloat16 else dt
            B_bf16 = B.bfloat16() if B.dtype != torch.bfloat16 else B
            x_bf16 = x.bfloat16() if x.dtype != torch.bfloat16 else x

            A_bar, u_bar = cayley_fused(alpha_bf16, omega_bf16, dt_bf16, B_bf16, x_bf16, eps)
        else:
            # Use PyTorch reference
            A_bar, u_bar = cayley_fused_pytorch(alpha, omega, dt, B, x, eps)
            A_bar = A_bar.to(alpha.dtype)
            u_bar = u_bar.to(alpha.dtype)

        # Save for backward
        ctx.save_for_backward(alpha, omega, dt, B, x)
        ctx.eps = eps

        return A_bar, u_bar

    @staticmethod
    def backward(ctx, grad_A_bar: Tensor, grad_u_bar: Tensor) -> tuple[Tensor | None, ...]:
        """Backward pass using PyTorch (Phase 2).

        Computes gradients for alpha, omega, dt, B, x.
        Uses torch.autograd.grad with explicit enable_grad context.
        """
        alpha, omega, dt, B, x = ctx.saved_tensors
        eps = ctx.eps

        # Ensure gradients are float32 for precision
        grad_A_bar_float = grad_A_bar.float() if grad_A_bar.dtype == torch.bfloat16 else grad_A_bar
        grad_u_bar_float = grad_u_bar.float() if grad_u_bar.dtype == torch.bfloat16 else grad_u_bar

        # Use torch.enable_grad() to ensure autograd is enabled inside backward
        with torch.enable_grad():
            # Create leaf tensors that require gradients
            alpha_grad = alpha.detach().float().requires_grad_(True)
            omega_grad = omega.detach().float().requires_grad_(True)
            dt_grad = dt.detach().float().requires_grad_(True)
            B_grad = B.detach().float().requires_grad_(True)
            x_grad = x.detach().float().requires_grad_(True)

            # Recompute forward in float32 with autograd
            A_bar_recompute, u_bar_recompute = cayley_fused_pytorch(
                alpha_grad, omega_grad, dt_grad, B_grad, x_grad, eps
            )

            # Compute gradients using vector-Jacobian product
            grads = torch.autograd.grad(
                outputs=[A_bar_recompute, u_bar_recompute],
                inputs=[alpha_grad, omega_grad, dt_grad, B_grad, x_grad],
                grad_outputs=[grad_A_bar_float, grad_u_bar_float],
                allow_unused=True,
            )

        d_alpha, d_omega, d_dt, d_B, d_x = grads

        # Cast back to input dtype
        if d_alpha is not None:
            d_alpha = d_alpha.to(alpha.dtype)
        if d_omega is not None:
            d_omega = d_omega.to(omega.dtype)
        if d_dt is not None:
            d_dt = d_dt.to(dt.dtype)
        if d_B is not None:
            d_B = d_B.to(B.dtype)
        if d_x is not None:
            d_x = d_x.to(x.dtype)

        return d_alpha, d_omega, d_dt, d_B, d_x, None, None


def cayley_transform(
    alpha: Tensor,
    omega: Tensor,
    dt: Tensor,
    B: Tensor,
    x: Tensor,
    eps: float = 1e-6,
    use_triton: bool = True,
) -> tuple[Tensor, Tensor]:
    """Compute Cayley transform discretization with autograd support.

    This is the main entry point for the Cayley transform.

    Args:
        alpha: Damping coefficients (batch, seq, d_inner). Must be >= 0.
        omega: Frequency coefficients (batch, seq, d_inner).
        dt: Timestep (batch, seq, d_inner). Must be > 0.
        B: Input projection (batch, seq, d_inner, 2).
        x: Input values (batch, seq, d_inner).
        eps: Numerical stability epsilon.
        use_triton: Whether to use Triton kernel.

    Returns:
        A_bar: Discretized transition (batch, seq, d_inner, 4).
        u_bar: Discretized input (batch, seq, d_inner, 2).

    Example:
        >>> alpha = torch.rand(8, 256, 64, device='cuda')
        >>> omega = torch.randn(8, 256, 64, device='cuda')
        >>> dt = torch.ones(8, 256, 64, device='cuda') * 0.01
        >>> B = torch.randn(8, 256, 64, 2, device='cuda')
        >>> x = torch.randn(8, 256, 64, device='cuda')
        >>> A_bar, u_bar = cayley_transform(alpha, omega, dt, B, x)
    """
    return CayleyTransformOp.apply(alpha, omega, dt, B, x, eps, use_triton)


def cayley_transform_no_grad(
    alpha: Tensor,
    omega: Tensor,
    dt: Tensor,
    B: Tensor,
    x: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Compute Cayley transform without gradient tracking (inference).

    Directly calls Triton kernel without autograd overhead.
    """
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

    alpha_bf16 = alpha.bfloat16() if alpha.dtype != torch.bfloat16 else alpha
    omega_bf16 = omega.bfloat16() if omega.dtype != torch.bfloat16 else omega
    dt_bf16 = dt.bfloat16() if dt.dtype != torch.bfloat16 else dt
    B_bf16 = B.bfloat16() if B.dtype != torch.bfloat16 else B
    x_bf16 = x.bfloat16() if x.dtype != torch.bfloat16 else x

    return cayley_fused(alpha_bf16, omega_bf16, dt_bf16, B_bf16, x_bf16, eps)
