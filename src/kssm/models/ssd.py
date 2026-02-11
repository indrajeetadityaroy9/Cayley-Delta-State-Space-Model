"""State Space Duality (SSD) - Chunkwise Parallel KSSM with Delta-Rule.

This module implements the chunkwise parallel scan algorithm (Mamba-2 style),
adapted for the 2x2 Cayley dynamics of KSSM with gated delta-rule state updates.

Algorithm:
    1.  Split sequence into chunks of size Q (e.g., 64).
    2.  Intra-Chunk (Local): Sequential delta-rule scan assuming h_{-1}=0.
        h[t] = A_bar[t] @ (h[t-1] - beta*kTh*k) + beta*v*k^T
        Also tracks cumulative base A_bar operators (cum_A) for each chunk.
    3.  Inter-Chunk (Global): Recurrently update state h between chunks using
        the cumulative base transition operators.
    4.  Combine: Correct local states using the propagated global chunk states.

The delta-rule erasure (step 2) enables selective forgetting of specific
associations without uniform decay. Inter-chunk correction uses base A_bar
products (without erasure), which is an acceptable approximation since most
associations form and are erased within a single chunk of 64 steps.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Constants for chunking
CHUNK_SIZE = 64


@torch.compile
def _intra_chunk_scan_delta(
    A_flat: Tensor,     # (B*NC, C, H, 2, 2)
    K_flat: Tensor,     # (B*NC, C, H, D, 2)
    V_flat: Tensor,     # (B*NC, C, H, D, 1)
    beta_flat: Tensor,  # (B*NC, C, H)
    dt_flat: Tensor,    # (B*NC, C, H)
):
    """
    Sequential delta-rule intra-chunk scan with Cayley dynamics.

    For each timestep t within a chunk:
        kTh = dot(k[t], h[t-1])                 # per-dim scalar
        h_mod = h[t-1] - beta[t] * kTh * k[t]   # selective erasure
        h[t] = A_bar[t] @ h_mod + beta[t] * (v[t] * k[t]^T) * dt[t]  # evolve + inject

    Also tracks cumulative A_bar product (per-head, not per-dim) for
    inter-chunk correction.

    Returns:
        local_h: (B*NC, C, H, D, 2) — state at each position
        cum_A:   (B*NC, C, H, 2, 2) — cumulative A_bar products
    """
    batch_chunks, C, H, D, _ = K_flat.shape
    device = K_flat.device
    dtype = K_flat.dtype

    local_h = torch.empty(batch_chunks, C, H, D, 2, device=device, dtype=dtype)
    cum_A = torch.empty(batch_chunks, C, H, 2, 2, device=device, dtype=dtype)

    # State in FP32 for numerical stability
    h = torch.zeros(batch_chunks, H, D, 2, device=device, dtype=torch.float32)

    # Cumulative A_bar product (per-head 2x2)
    curr_A = torch.eye(2, device=device, dtype=torch.float32)
    curr_A = curr_A.view(1, 1, 2, 2).expand(batch_chunks, H, 2, 2).contiguous()

    for t in range(C):
        A_t = A_flat[:, t].float()          # (B*NC, H, 2, 2)
        k_t = K_flat[:, t].float()          # (B*NC, H, D, 2)
        v_t = V_flat[:, t].float()          # (B*NC, H, D, 1)
        b_t = beta_flat[:, t].float()       # (B*NC, H)
        dt_t = dt_flat[:, t].float()        # (B*NC, H)

        # Delta-rule selective erasure
        # kTh: dot product of k[t] and h over state dim (last dim, size 2)
        # k_t: (B*NC, H, D, 2), h: (B*NC, H, D, 2) -> kTh: (B*NC, H, D)
        kTh = (k_t * h).sum(dim=-1)        # (B*NC, H, D)

        # h_mod = h - beta * kTh * k  (per-dim erasure)
        # b_t: (B*NC, H) -> expand to (B*NC, H, D, 1)
        b_t_exp = b_t.unsqueeze(-1).unsqueeze(-1)  # (B*NC, H, 1, 1)
        h_mod = h - b_t_exp * kTh.unsqueeze(-1) * k_t  # (B*NC, H, D, 2)

        # Apply A_bar rotation-damping (per-head, broadcast over D)
        # A_t: (B*NC, H, 2, 2), h_mod: (B*NC, H, D, 2)
        h_evolved = torch.einsum("bhij,bhdj->bhdi", A_t, h_mod)

        # Delta-rule injection: beta * v * k^T * dt
        # v_t: (B*NC, H, D, 1), k_t: (B*NC, H, D, 2) -> outer: (B*NC, H, D, 2)
        dt_t_exp = dt_t.unsqueeze(-1).unsqueeze(-1)  # (B*NC, H, 1, 1)
        injection = b_t_exp * v_t * k_t * dt_t_exp   # (B*NC, H, D, 2)

        h = h_evolved + injection

        local_h[:, t] = h.to(dtype)

        # Track cumulative base A_bar product (for inter-chunk correction)
        curr_A = torch.matmul(A_t, curr_A)
        cum_A[:, t] = curr_A.to(dtype)

    return local_h, cum_A


@torch.compile
def _inter_chunk_scan(
    total_A: Tensor,       # (B, NC, H, 2, 2)
    final_local_h: Tensor, # (B, NC, H, D, 2)
):
    """
    Sequential recurrence across chunks.
    """
    B, n_chunks, H, D, _ = final_local_h.shape
    device = total_A.device
    dtype = total_A.dtype

    chunk_states = torch.empty(B, n_chunks, H, D, 2, device=device, dtype=dtype)

    # Use FP32 for state accumulation to prevent swamping
    state = torch.zeros(B, H, D, 2, device=device, dtype=torch.float32)

    total_A_fp32 = total_A.float()
    final_local_h_fp32 = final_local_h.float()

    for k in range(n_chunks):
        chunk_states[:, k] = state.to(dtype)

        # h_{k+1_in} = total_A_k @ h_{k_in} + final_local_h_k
        state = torch.einsum("bhij,bhdj->bhdi", total_A_fp32[:, k], state) + final_local_h_fp32[:, k]

    return chunk_states


class SSDChunkwiseScan(nn.Module):
    """Chunkwise Parallel Scan for KSSM with delta-rule state updates."""

    def __init__(self, d_inner: int, n_heads: int, chunk_size: int = CHUNK_SIZE):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.head_dim = d_inner // n_heads

    def forward(
        self,
        alpha: Tensor,
        omega: Tensor,
        dt: Tensor,
        K: Tensor,
        V: Tensor,
        beta: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            alpha, omega, dt: (B, L, H)
            K: (B, L, H, D, 2) - reshaped per head
            V: (B, L, H, D, 1) - reshaped per head
            beta: (B, L, H) - delta-rule update gate, or None for Hebbian fallback

        Returns:
            Y: (B, L, H, D, 2) - evolved states
            chunk_states: (B, NC, H, D, 2) - inter-chunk states for feedback
        """
        B, L, H, D, _ = K.shape

        # 1. Discretize Dynamics (Parallel)
        # A = [[-alpha, omega], [-omega, -alpha]]
        tau = dt * 0.5
        tau_alpha = tau * alpha
        tau_omega = tau * omega
        denom = (1.0 + tau_alpha).pow(2) + tau_omega.pow(2) + 1e-6
        inv_det = 1.0 / denom

        # A_bar = (I - tau*A)^{-1}(I + tau*A)
        a11 = ((1.0 + tau_alpha) * (1.0 - tau_alpha) - tau_omega.pow(2)) * inv_det
        a12 = (2.0 * tau_omega) * inv_det

        # A_bar: (B, L, H, 2, 2)
        A_bar = torch.stack([
            torch.stack([a11, a12], dim=-1),
            torch.stack([-a12, a11], dim=-1)
        ], dim=-2)

        # Default beta to 1.0 (standard injection, no erasure) if not provided
        if beta is None:
            beta = torch.ones(B, L, H, device=K.device, dtype=K.dtype)

        # 2. Chunk Preparation
        n_chunks = math.ceil(L / self.chunk_size)
        pad_len = n_chunks * self.chunk_size - L

        if pad_len > 0:
            A_bar = F.pad(A_bar, (0, 0, 0, 0, 0, 0, 0, pad_len))
            K = F.pad(K, (0, 0, 0, 0, 0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, 0, 0, 0, 0, pad_len))
            beta = F.pad(beta, (0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, pad_len))

        # Reshape to (B, NC, C, H, ...)
        C = self.chunk_size
        A_chunk = A_bar.view(B, n_chunks, C, H, 2, 2)
        K_chunk = K.view(B, n_chunks, C, H, D, 2)
        V_chunk = V.view(B, n_chunks, C, H, D, 1)
        beta_chunk = beta.view(B, n_chunks, C, H)
        dt_chunk = dt.view(B, n_chunks, C, H)

        # Flatten for batched intra-chunk scan
        A_flat = A_chunk.reshape(B * n_chunks, C, H, 2, 2)
        K_flat = K_chunk.reshape(B * n_chunks, C, H, D, 2)
        V_flat = V_chunk.reshape(B * n_chunks, C, H, D, 1)
        beta_flat = beta_chunk.reshape(B * n_chunks, C, H)
        dt_flat = dt_chunk.reshape(B * n_chunks, C, H)

        # === 3. Intra-Chunk Scan (Local, Delta-Rule) ===
        local_h, cum_A = _intra_chunk_scan_delta(A_flat, K_flat, V_flat, beta_flat, dt_flat)

        # Reshape back to (B, NC, ...)
        local_h = local_h.view(B, n_chunks, C, H, D, 2)
        cum_A = cum_A.view(B, n_chunks, C, H, 2, 2)

        # Extract summaries for inter-chunk recurrence
        total_A = cum_A[:, :, -1]         # (B, NC, H, 2, 2)
        final_local_h = local_h[:, :, -1] # (B, NC, H, D, 2)

        # === 4. Inter-Chunk Recurrence (Global) ===
        chunk_states = _inter_chunk_scan(total_A, final_local_h)

        # === 5. Correction (Broadcast) ===
        # True h_{t} = local_h_{t} + (cum_A_{t} @ chunk_state_{k})
        correction = torch.einsum("bnchij,bnhdj->bnchdi", cum_A, chunk_states)

        Y = local_h + correction

        # Reshape to flat sequence (B, L_pad, H, D, 2)
        Y = Y.reshape(B, -1, H, D, 2)

        # Unpad
        if pad_len > 0:
            Y = Y[:, :L]

        return Y, chunk_states
