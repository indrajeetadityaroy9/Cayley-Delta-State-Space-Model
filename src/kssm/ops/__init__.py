"""KSSM fused CUDA operations with autograd support."""

import torch
from torch import Tensor
from torch.autograd import Function

import kssm._C as _C


# ============================================================================
# Conv1d + SiLU (existing)
# ============================================================================

class CUDAConv1dSiLUOp(Function):
    """PyTorch autograd Function for fused Conv1d + SiLU CUDA kernel."""

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        out = _C.conv1d_silu_fwd_cuda(x, weight, bias)
        ctx.save_for_backward(x, weight, bias)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x, weight, bias = ctx.saved_tensors
        grad_x, grad_weight, grad_bias = _C.conv1d_silu_bwd_cuda(
            x, weight, bias, grad_out
        )
        return grad_x, grad_weight, grad_bias


def conv1d_silu_cuda(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return CUDAConv1dSiLUOp.apply(x, weight, bias)


# ============================================================================
# Intra-Chunk Delta-Rule Scan
# ============================================================================

class IntraChunkScanFn(Function):
    """Fused intra-chunk delta-rule scan with Cayley dynamics."""

    @staticmethod
    def forward(
        ctx,
        A_flat: Tensor,     # (BNC, C, H, 2, 2) BF16
        K_flat: Tensor,     # (BNC, C, H, D)    BF16
        V_flat: Tensor,     # (BNC, C, H, 2)    BF16
        beta_flat: Tensor,  # (BNC, C, H)       BF16
    ) -> tuple[Tensor, Tensor]:
        local_h, cum_A = _C.intra_chunk_scan_fwd_cuda(
            A_flat, K_flat, V_flat, beta_flat
        )
        ctx.save_for_backward(A_flat, K_flat, V_flat, beta_flat, local_h, cum_A)
        return local_h, cum_A

    @staticmethod
    def backward(
        ctx, grad_local_h: Tensor, grad_cum_A: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        A_flat, K_flat, V_flat, beta_flat, local_h, cum_A = ctx.saved_tensors
        grad_A, grad_K, grad_V, grad_beta = _C.intra_chunk_scan_bwd_cuda(
            grad_local_h, grad_cum_A,
            A_flat, K_flat, V_flat, beta_flat,
            local_h, cum_A,
        )
        # grad_beta comes back as FP32, convert to match input dtype
        grad_beta = grad_beta.to(beta_flat.dtype)
        return grad_A, grad_K, grad_V, grad_beta


def intra_chunk_scan_cuda(
    A_flat: Tensor, K_flat: Tensor, V_flat: Tensor, beta_flat: Tensor
) -> tuple[Tensor, Tensor]:
    """Fused intra-chunk delta-rule scan.

    Args:
        A_flat: (BNC, C, H, 2, 2) — pre-computed A_bar matrices
        K_flat: (BNC, C, H, D) — keys
        V_flat: (BNC, C, H, 2) — values (2D Hamiltonian)
        beta_flat: (BNC, C, H) — delta-rule update gate

    Returns:
        local_h: (BNC, C, H, 2, D) — state at each position
        cum_A: (BNC, C, H, 2, 2) — cumulative A_bar products
    """
    return IntraChunkScanFn.apply(A_flat, K_flat, V_flat, beta_flat)


# ============================================================================
# Inter-Chunk Sequential Scan
# ============================================================================

class InterChunkScanFn(Function):
    """Sequential recurrence across chunks."""

    @staticmethod
    def forward(
        ctx,
        total_A: Tensor,        # (B, NC, H, 2, 2)
        final_local_h: Tensor,  # (B, NC, H, 2, D)
    ) -> Tensor:
        chunk_states = _C.inter_chunk_scan_fwd_cuda(total_A, final_local_h)
        ctx.save_for_backward(total_A, chunk_states)
        return chunk_states

    @staticmethod
    def backward(ctx, grad_chunk_states: Tensor) -> tuple[Tensor, Tensor]:
        total_A, chunk_states = ctx.saved_tensors
        grad_total_A, grad_final_local_h = _C.inter_chunk_scan_bwd_cuda(
            grad_chunk_states, total_A, chunk_states
        )
        return grad_total_A, grad_final_local_h


def inter_chunk_scan_cuda(
    total_A: Tensor, final_local_h: Tensor
) -> Tensor:
    """Sequential inter-chunk state propagation.

    Args:
        total_A: (B, NC, H, 2, 2) — cumulative A_bar per chunk
        final_local_h: (B, NC, H, 2, D) — final local state per chunk

    Returns:
        chunk_states: (B, NC, H, 2, D) — propagated state entering each chunk
    """
    return InterChunkScanFn.apply(total_A, final_local_h)


# ============================================================================
# Fused Cayley Discretization + VP Scale
# ============================================================================

class CayleyVPFn(Function):
    """Fused Cayley discretization + recurrence gate + variance-preserving scale."""

    @staticmethod
    def forward(
        ctx,
        alpha: Tensor,    # (B, L, H)
        omega: Tensor,    # (B, L, H)
        dt: Tensor,       # (B, L, H)
        r_gate: Tensor,   # (B, L, H) or empty tensor
        gating_c: float,
    ) -> tuple[Tensor, Tensor]:
        A_bar, vp_scale = _C.cayley_vp_fwd_cuda(alpha, omega, dt, r_gate, gating_c)
        ctx.save_for_backward(alpha, omega, dt, r_gate)
        ctx.gating_c = gating_c
        return A_bar, vp_scale

    @staticmethod
    def backward(
        ctx, grad_A_bar: Tensor, grad_vp_scale: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, None]:
        alpha, omega, dt, r_gate = ctx.saved_tensors
        grad_alpha, grad_omega, grad_dt, grad_r = _C.cayley_vp_bwd_cuda(
            grad_A_bar, grad_vp_scale,
            alpha, omega, dt, r_gate,
            ctx.gating_c,
        )
        # Convert FP32 grads back to input dtype
        input_dtype = alpha.dtype
        grad_alpha = grad_alpha.to(input_dtype)
        grad_omega = grad_omega.to(input_dtype)
        grad_dt = grad_dt.to(input_dtype)
        grad_r_out = grad_r.to(input_dtype) if r_gate.numel() > 0 else None
        return grad_alpha, grad_omega, grad_dt, grad_r_out, None


def cayley_vp_cuda(
    alpha: Tensor, omega: Tensor, dt: Tensor,
    r_gate: Tensor, gating_c: float,
) -> tuple[Tensor, Tensor]:
    """Fused Cayley discretization + recurrence gate modulation + VP scale.

    Args:
        alpha, omega, dt: (B, L, H) dynamics parameters
        r_gate: (B, L, H) recurrence gate, or empty tensor if unused
        gating_c: gating range constant

    Returns:
        A_bar: (B, L, H, 2, 2) — discretized transition matrices
        vp_scale: (B, L, H) — variance-preserving scale
    """
    return CayleyVPFn.apply(alpha, omega, dt, r_gate, gating_c)


# ============================================================================
# Adaptive Timestep
# ============================================================================

class AdaptiveDtFn(Function):
    """Fused adaptive timestep computation."""

    @staticmethod
    def forward(
        ctx,
        alpha: Tensor,         # (B, L, H)
        omega: Tensor,         # (B, L, H)
        log_dt_scale: Tensor,  # (H,) FP32
        omega_thresh: float,
        delta: float,
        smoothness: float,
        eps: float,
    ) -> Tensor:
        dt_out = _C.adaptive_dt_fwd_cuda(
            alpha, omega, log_dt_scale,
            omega_thresh, delta, smoothness, eps,
        )
        ctx.save_for_backward(alpha, omega, log_dt_scale)
        ctx.omega_thresh = omega_thresh
        ctx.delta = delta
        ctx.smoothness = smoothness
        ctx.eps = eps
        return dt_out

    @staticmethod
    def backward(ctx, grad_dt: Tensor) -> tuple[Tensor, Tensor, Tensor, None, None, None, None]:
        alpha, omega, log_dt_scale = ctx.saved_tensors
        grad_alpha, grad_omega, grad_log_dt_scale = _C.adaptive_dt_bwd_cuda(
            grad_dt, alpha, omega, log_dt_scale,
            ctx.omega_thresh, ctx.delta, ctx.smoothness, ctx.eps,
        )
        input_dtype = alpha.dtype
        return (
            grad_alpha.to(input_dtype),
            grad_omega.to(input_dtype),
            grad_log_dt_scale,  # stays FP32 (parameter)
            None, None, None, None,
        )


def adaptive_dt_cuda(
    alpha: Tensor, omega: Tensor, log_dt_scale: Tensor,
    omega_thresh: float, delta: float, smoothness: float, eps: float,
) -> Tensor:
    """Fused adaptive timestep.

    Args:
        alpha, omega: (B, L, H) dynamics parameters
        log_dt_scale: (H,) learned per-head scale (FP32 parameter)
        omega_thresh, delta, smoothness, eps: safety constants

    Returns:
        dt: (B, L, H) adaptive timestep
    """
    return AdaptiveDtFn.apply(
        alpha, omega, log_dt_scale,
        omega_thresh, delta, smoothness, eps,
    )


__all__ = [
    "conv1d_silu_cuda",
    "intra_chunk_scan_cuda",
    "inter_chunk_scan_cuda",
    "cayley_vp_cuda",
    "adaptive_dt_cuda",
]
