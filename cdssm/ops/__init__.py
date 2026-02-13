"""CDSSM fused CUDA operations with autograd support."""

import torch
from torch import Tensor
from torch.autograd import Function

import cdssm._C as _C


# Conv1d + SiLU (existing)

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


# Intra-Chunk Delta-Rule Scan

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


# Inter-Chunk Sequential Scan

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


# Fused Cayley Discretization + VP Scale

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


# Adaptive Timestep

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


# Fused Dynamics Pipeline (Phase 3)

class DynamicsFusedFn(Function):
    """Fused dynamics: gate_raw → A_bar, vp_scale, beta, sel_C_gate in one kernel."""

    @staticmethod
    def forward(
        ctx,
        gate_raw: Tensor,        # (B, L, 7*H) BF16
        log_dt_scale: Tensor,    # (H,) FP32
        rope_freqs: Tensor,      # (H,) FP32
        gating_c: float,
        omega_thresh: float,
        adt_delta: float,
        adt_smoothness: float,
        adt_eps: float,
        n_heads: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        A_bar, vp_scale, beta, sel_C_gate = _C.dynamics_fused_fwd_cuda(
            gate_raw, log_dt_scale, rope_freqs,
            gating_c, omega_thresh, adt_delta, adt_smoothness, adt_eps,
            n_heads,
        )
        ctx.save_for_backward(gate_raw, log_dt_scale, rope_freqs)
        ctx.gating_c = gating_c
        ctx.omega_thresh = omega_thresh
        ctx.adt_delta = adt_delta
        ctx.adt_smoothness = adt_smoothness
        ctx.adt_eps = adt_eps
        ctx.n_heads = n_heads
        return A_bar, vp_scale, beta, sel_C_gate

    @staticmethod
    def backward(
        ctx, grad_A_bar: Tensor, grad_vp_scale: Tensor,
        grad_beta: Tensor, grad_sel_C_gate: Tensor,
    ) -> tuple[Tensor, Tensor, None, None, None, None, None, None, None]:
        gate_raw, log_dt_scale, rope_freqs = ctx.saved_tensors
        grad_gate_raw, grad_log_dt_scale = _C.dynamics_fused_bwd_cuda(
            grad_A_bar, grad_vp_scale, grad_beta, grad_sel_C_gate,
            gate_raw, log_dt_scale, rope_freqs,
            ctx.gating_c, ctx.omega_thresh,
            ctx.adt_delta, ctx.adt_smoothness, ctx.adt_eps,
            ctx.n_heads,
        )
        return grad_gate_raw, grad_log_dt_scale, None, None, None, None, None, None, None


def dynamics_fused_cuda(
    gate_raw: Tensor, log_dt_scale: Tensor, rope_freqs: Tensor,
    gating_c: float, omega_thresh: float, adt_delta: float,
    adt_smoothness: float, adt_eps: float, n_heads: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fused dynamics pipeline: gate_raw → (A_bar, vp_scale, beta, sel_C_gate).

    Replaces adaptive_dt_cuda + cayley_vp_cuda + ~15 elementwise PyTorch ops.

    Args:
        gate_raw: (B, L, 7*H) raw gate projection output (BF16)
        log_dt_scale: (H,) learned per-head scale (FP32)
        rope_freqs: (H,) RoPE frequencies (FP32)
        gating_c, omega_thresh, adt_delta, adt_smoothness, adt_eps: scalar constants
        n_heads: number of attention heads

    Returns:
        A_bar: (B, L, H, 2, 2) discretized transition matrices
        vp_scale: (B, L, H) variance-preserving scale
        beta: (B, L, H) fused write gate = sigmoid(beta_raw) * sigmoid(sel_B)
        sel_C_gate: (B, L, H) read gate = sigmoid(sel_C)
    """
    return DynamicsFusedFn.apply(
        gate_raw, log_dt_scale, rope_freqs,
        gating_c, omega_thresh, adt_delta, adt_smoothness, adt_eps, n_heads,
    )


# Fused K/Q L2 Normalization (Phase 4)

class NormalizeKQFn(Function):
    """Fused L2 normalization for K and Q vectors."""

    @staticmethod
    def forward(ctx, K: Tensor, Q: Tensor) -> tuple[Tensor, Tensor]:
        K_norm, Q_norm = _C.normalize_kq_fwd_cuda(K, Q)
        ctx.save_for_backward(K, Q)
        return K_norm, Q_norm

    @staticmethod
    def backward(ctx, grad_K_out: Tensor, grad_Q_out: Tensor) -> tuple[Tensor, Tensor]:
        K, Q = ctx.saved_tensors
        grad_K_in, grad_Q_in = _C.normalize_kq_bwd_cuda(grad_K_out, grad_Q_out, K, Q)
        return grad_K_in, grad_Q_in


def normalize_kq_cuda(K: Tensor, Q: Tensor) -> tuple[Tensor, Tensor]:
    """Fused L2 normalization for K and Q vectors.

    Args:
        K: (B, L, H, D) key vectors (BF16)
        Q: (B, L, H, D) query vectors (BF16)

    Returns:
        K_norm: (B, L, H, D) L2-normalized keys
        Q_norm: (B, L, H, D) L2-normalized queries
    """
    return NormalizeKQFn.apply(K, Q)


__all__ = [
    "conv1d_silu_cuda",
    "intra_chunk_scan_cuda",
    "inter_chunk_scan_cuda",
    "cayley_vp_cuda",
    "adaptive_dt_cuda",
    "dynamics_fused_cuda",
    "normalize_kq_cuda",
]
