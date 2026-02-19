"""CDSSM fused CUDA operations with autograd support.

Each autograd Function wraps a forward/backward kernel pair from cdssm._C
(compiled from cdssm/csrc/kernels/*.cu via pybind11).
"""

from torch import Tensor
from torch.autograd import Function

import cdssm._C as _C


class CUDAConv1dSiLUOp(Function):
    """Fused Conv1d + SiLU CUDA kernel."""

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


class DynamicsFusedFn(Function):
    """Fused dynamics: gate_raw -> A_bar, vp_scale, beta, sel_C_gate in one kernel.

    Gate layout: [alpha_0..alpha_{N/2-1}, omega_0..omega_{N/2-1},
                  sel_B, sel_C, sel_dt, beta, r_gate] x H
    Output: A_bar (B,L,H,N), vp_scale (B,L,H,N/2), beta (B,L,H), sel_C (B,L,H)
    """

    @staticmethod
    def forward(
        ctx,
        gate_raw: Tensor,        # (B, L, (N+5)*H) BF16
        log_dt_scale: Tensor,    # (H,) FP32
        rope_freqs: Tensor,      # (H,) FP32
        gating_c: float,
        omega_thresh: float,
        adt_delta: float,
        adt_smoothness: float,
        adt_eps: float,
        n_heads: int,
        state_dim: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        A_bar, vp_scale, beta, sel_C_gate = _C.dynamics_fused_fwd_cuda(
            gate_raw, log_dt_scale, rope_freqs,
            gating_c, omega_thresh, adt_delta, adt_smoothness, adt_eps,
            n_heads, state_dim,
        )
        ctx.save_for_backward(gate_raw, log_dt_scale, rope_freqs)
        ctx.gating_c = gating_c
        ctx.omega_thresh = omega_thresh
        ctx.adt_delta = adt_delta
        ctx.adt_smoothness = adt_smoothness
        ctx.adt_eps = adt_eps
        ctx.n_heads = n_heads
        ctx.state_dim = state_dim
        return A_bar, vp_scale, beta, sel_C_gate

    @staticmethod
    def backward(
        ctx, grad_A_bar: Tensor, grad_vp_scale: Tensor,
        grad_beta: Tensor, grad_sel_C_gate: Tensor,
    ) -> tuple[Tensor, Tensor, None, None, None, None, None, None, None, None]:
        gate_raw, log_dt_scale, rope_freqs = ctx.saved_tensors
        grad_gate_raw, grad_log_dt_scale = _C.dynamics_fused_bwd_cuda(
            grad_A_bar, grad_vp_scale, grad_beta, grad_sel_C_gate,
            gate_raw, log_dt_scale, rope_freqs,
            ctx.gating_c, ctx.omega_thresh,
            ctx.adt_delta, ctx.adt_smoothness, ctx.adt_eps,
            ctx.n_heads, ctx.state_dim,
        )
        return grad_gate_raw, grad_log_dt_scale, None, None, None, None, None, None, None, None


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


class IntraChunkScanFn(Function):
    """Fused intra-chunk delta-rule scan with complex diagonal Cayley dynamics.

    State h in R^(N x D) with N/2 complex eigenvalue pairs.
    A_bar stores conj(mu) as re/im interleaved.
    """

    @staticmethod
    def forward(
        ctx,
        A_flat: Tensor,     # (BNC, C, H, N) BF16, re/im interleaved
        K_flat: Tensor,     # (BNC, C, H, D) BF16
        V_flat: Tensor,     # (BNC, C, H, N) BF16
        beta_flat: Tensor,  # (BNC, C, H)    BF16
        state_dim: int,
    ) -> tuple[Tensor, Tensor]:
        local_h, cum_A = _C.intra_chunk_scan_fwd_cuda(
            A_flat, K_flat, V_flat, beta_flat, state_dim
        )
        ctx.save_for_backward(A_flat, K_flat, V_flat, beta_flat, local_h, cum_A)
        ctx.state_dim = state_dim
        return local_h, cum_A

    @staticmethod
    def backward(
        ctx, grad_local_h: Tensor, grad_cum_A: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, None]:
        A_flat, K_flat, V_flat, beta_flat, local_h, cum_A = ctx.saved_tensors
        grad_A, grad_K, grad_V, grad_beta = _C.intra_chunk_scan_bwd_cuda(
            grad_local_h, grad_cum_A,
            A_flat, K_flat, V_flat, beta_flat,
            local_h, cum_A,
            ctx.state_dim,
        )
        grad_beta = grad_beta.to(beta_flat.dtype)
        return grad_A, grad_K, grad_V, grad_beta, None


class InterChunkScanFn(Function):
    """Sequential complex diagonal recurrence across chunks.

    State s in R^N with N/2 complex eigenvalue pairs.
    total_A (B,NC,H,N) stores conj(mu) re/im interleaved.
    """

    @staticmethod
    def forward(
        ctx,
        total_A: Tensor,        # (B, NC, H, N)
        final_local_h: Tensor,  # (B, NC, H, N, D)
        state_dim: int,
    ) -> Tensor:
        chunk_states = _C.inter_chunk_scan_fwd_cuda(total_A, final_local_h, state_dim)
        ctx.save_for_backward(total_A, chunk_states)
        ctx.state_dim = state_dim
        return chunk_states

    @staticmethod
    def backward(ctx, grad_chunk_states: Tensor) -> tuple[Tensor, Tensor, None]:
        total_A, chunk_states = ctx.saved_tensors
        grad_total_A, grad_final_local_h = _C.inter_chunk_scan_bwd_cuda(
            grad_chunk_states, total_A, chunk_states, ctx.state_dim
        )
        return grad_total_A, grad_final_local_h, None


class ExactCorrectionFn(Function):
    """Fused exact correction: propagates chunk states through dynamics (erasure + rotation, no injection).

    Replaces the Python loop over C timesteps with a single CUDA kernel.
    """

    @staticmethod
    def forward(
        ctx,
        A_chunk: Tensor,       # (B, NC, C, H, N) BF16
        K_chunk: Tensor,       # (B, NC, C, H, D) BF16
        beta_chunk: Tensor,    # (B, NC, C, H)    BF16
        chunk_states: Tensor,  # (B, NC, H, N, D) BF16
        state_dim: int,
    ) -> Tensor:
        corrections = _C.exact_correction_fwd_cuda(
            A_chunk, K_chunk, beta_chunk, chunk_states, state_dim
        )
        ctx.save_for_backward(A_chunk, K_chunk, beta_chunk, chunk_states, corrections)
        ctx.state_dim = state_dim
        return corrections

    @staticmethod
    def backward(ctx, grad_corrections: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, None]:
        A_chunk, K_chunk, beta_chunk, chunk_states, corrections = ctx.saved_tensors
        grad_A, grad_K, grad_beta, grad_states = _C.exact_correction_bwd_cuda(
            grad_corrections, A_chunk, K_chunk, beta_chunk,
            chunk_states, corrections, ctx.state_dim
        )
        return grad_A, grad_K, grad_beta, grad_states, None
