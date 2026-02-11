"""State Space Duality (SSD) - Chunkwise Parallel KSSM with Delta-Rule.

This module implements the chunkwise parallel scan algorithm (Mamba-2 style),
adapted for the 2x2 Cayley dynamics of KSSM with gated delta-rule state updates.

Algorithm:
    1.  Split sequence into chunks of size Q (e.g., 64).
    2.  Intra-Chunk (Local): Sequential delta-rule scan assuming h_{-1}=0.
        State is a (2, D) matrix memory per head. Keys are D-dimensional,
        values are 2-dimensional (Hamiltonian state vectors).
        h[t] = A_bar[t] @ (h[t-1] - beta*(h@k)*k^T) + beta*(v*k^T)*dt
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
CHUNK_SIZE = 64  # = head_dim, aligned to H100 Tensor Core width


@torch.compile
def _intra_chunk_scan_delta(
    A_flat: Tensor,     # (B*NC, C, H, 2, 2)
    K_flat: Tensor,     # (B*NC, C, H, D)
    V_flat: Tensor,     # (B*NC, C, H, 2)
    beta_flat: Tensor,  # (B*NC, C, H)
    dt_flat: Tensor,    # (B*NC, C, H)
):
    """
    Sequential delta-rule intra-chunk scan with Cayley dynamics and matrix memory.

    State h is (H, 2, D) per batch element — a matrix memory where keys are
    D-dimensional and values are 2-dimensional Hamiltonian state vectors.

    For each timestep t within a chunk:
        kTh = h @ k          -> (B*NC, H, 2) retrieval
        h_mod = h - beta * (kTh outer k)  -> selective erasure
        h[t] = A @ h_mod + beta * (v outer k) * dt  -> evolve + inject

    Returns:
        local_h: (B*NC, C, H, 2, D) — state at each position
        cum_A:   (B*NC, C, H, 2, 2) — cumulative A_bar products
    """
    batch_chunks, C, H, D = K_flat.shape
    device = K_flat.device
    dtype = K_flat.dtype

    local_h = torch.empty(batch_chunks, C, H, 2, D, device=device, dtype=dtype)
    cum_A = torch.empty(batch_chunks, C, H, 2, 2, device=device, dtype=dtype)

    # State in FP32 for numerical stability: (B*NC, H, 2, D)
    h = torch.zeros(batch_chunks, H, 2, D, device=device, dtype=torch.float32)

    # Cumulative A_bar product (per-head 2x2)
    curr_A = torch.eye(2, device=device, dtype=torch.float32)
    curr_A = curr_A.view(1, 1, 2, 2).expand(batch_chunks, H, 2, 2).contiguous()

    for t in range(C):
        A_t = A_flat[:, t].float()          # (B*NC, H, 2, 2)
        k_t = K_flat[:, t].float()          # (B*NC, H, D)
        v_t = V_flat[:, t].float()          # (B*NC, H, 2)
        b_t = beta_flat[:, t].float()       # (B*NC, H)
        dt_t = dt_flat[:, t].float()        # (B*NC, H)

        # Delta-rule retrieval: h @ k -> 2-vector per head
        # h: (B*NC, H, 2, D), k_t: (B*NC, H, D) -> kTh: (B*NC, H, 2)
        kTh = torch.einsum('bhsd,bhd->bhs', h, k_t)

        # Selective erasure: h -= beta * (kTh outer k)
        b_t_exp = b_t.unsqueeze(-1).unsqueeze(-1)  # (B*NC, H, 1, 1)
        erasure = b_t_exp * torch.einsum('bhs,bhd->bhsd', kTh, k_t)  # (B*NC, H, 2, D)
        h_mod = h - erasure

        # Apply A_bar rotation-damping (per-head 2x2, broadcast over D)
        # A_t: (B*NC, H, 2, 2), h_mod: (B*NC, H, 2, D)
        h_evolved = torch.einsum('bhij,bhjd->bhid', A_t, h_mod)

        # Delta-rule injection: beta * (v outer k) * dt
        # v_t: (B*NC, H, 2), k_t: (B*NC, H, D) -> outer: (B*NC, H, 2, D)
        dt_t_exp = dt_t.unsqueeze(-1).unsqueeze(-1)  # (B*NC, H, 1, 1)
        injection = b_t_exp * torch.einsum('bhs,bhd->bhsd', v_t, k_t) * dt_t_exp

        h = h_evolved + injection

        local_h[:, t] = h.to(dtype)

        # Track cumulative base A_bar product (for inter-chunk correction)
        curr_A = torch.matmul(A_t, curr_A)
        cum_A[:, t] = curr_A.to(dtype)

    return local_h, cum_A


@torch.compile
def _inter_chunk_scan(
    total_A: Tensor,       # (B, NC, H, 2, 2)
    final_local_h: Tensor, # (B, NC, H, 2, D)
):
    """
    Sequential recurrence across chunks with matrix state (H, 2, D).
    """
    B, n_chunks, H, _, D = final_local_h.shape
    device = total_A.device
    dtype = total_A.dtype

    chunk_states = torch.empty(B, n_chunks, H, 2, D, device=device, dtype=dtype)

    # Use FP32 for state accumulation
    state = torch.zeros(B, H, 2, D, device=device, dtype=torch.float32)

    total_A_fp32 = total_A.float()
    final_local_h_fp32 = final_local_h.float()

    for k in range(n_chunks):
        chunk_states[:, k] = state.to(dtype)

        # h_{k+1_in} = total_A_k @ h_{k_in} + final_local_h_k
        state = torch.einsum('bhij,bhjd->bhid', total_A_fp32[:, k], state) + final_local_h_fp32[:, k]

    return chunk_states


class SSDChunkwiseScan(nn.Module):
    """Chunkwise Parallel Scan for KSSM with delta-rule state updates."""

    def __init__(self, d_inner: int, n_heads: int, chunk_size: int = CHUNK_SIZE, gating_c: float = 8.0):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.head_dim = d_inner // n_heads
        self.gating_c = gating_c

    def forward(
        self,
        alpha: Tensor,
        omega: Tensor,
        dt: Tensor,
        K: Tensor,
        V: Tensor,
        beta: Tensor,
        r_gate: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            alpha, omega, dt: (B, L, H)
            K: (B, L, H, D) - D-dimensional keys per head
            V: (B, L, H, 2) - 2D Hamiltonian value vectors per head
            beta: (B, L, H) - delta-rule update gate (required)
            r_gate: (B, L, H) - recurrence gate for eigenvalue modulation (optional)

        Returns:
            Y: (B, L, H, 2, D) - evolved matrix states
            chunk_states: (B, NC, H, 2, D) - inter-chunk states for feedback
        """
        B, L, H, D = K.shape

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

        # Step 2: Recurrence gate modulates A_bar eigenvalue magnitude
        if r_gate is not None:
            c = self.gating_c
            numer = (1.0 - tau_alpha).square() + tau_omega.square()
            denom_e = (1.0 + tau_alpha).square() + tau_omega.square()
            eig_sq = numer / (denom_e + 1e-6)
            exponent = (c * r_gate - 1.0) / 2.0
            scale = eig_sq.clamp(min=1e-8).pow(exponent)
            A_bar = A_bar * scale.unsqueeze(-1).unsqueeze(-1)

        # 2. Chunk Preparation
        n_chunks = math.ceil(L / self.chunk_size)
        pad_len = n_chunks * self.chunk_size - L

        if pad_len > 0:
            A_bar = F.pad(A_bar, (0, 0, 0, 0, 0, 0, 0, pad_len))
            K = F.pad(K, (0, 0, 0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, 0, 0, pad_len))
            beta = F.pad(beta, (0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, pad_len))

        # Reshape to (B, NC, C, H, ...)
        C = self.chunk_size
        A_chunk = A_bar.view(B, n_chunks, C, H, 2, 2)
        K_chunk = K.view(B, n_chunks, C, H, D)
        V_chunk = V.view(B, n_chunks, C, H, 2)
        beta_chunk = beta.view(B, n_chunks, C, H)
        dt_chunk = dt.view(B, n_chunks, C, H)

        # Flatten for batched intra-chunk scan
        A_flat = A_chunk.reshape(B * n_chunks, C, H, 2, 2)
        K_flat = K_chunk.reshape(B * n_chunks, C, H, D)
        V_flat = V_chunk.reshape(B * n_chunks, C, H, 2)
        beta_flat = beta_chunk.reshape(B * n_chunks, C, H)
        dt_flat = dt_chunk.reshape(B * n_chunks, C, H)

        # === 3. Intra-Chunk Scan (Local, Delta-Rule) ===
        local_h, cum_A = _intra_chunk_scan_delta(A_flat, K_flat, V_flat, beta_flat, dt_flat)

        # Reshape back to (B, NC, ...)
        local_h = local_h.view(B, n_chunks, C, H, 2, D)
        cum_A = cum_A.view(B, n_chunks, C, H, 2, 2)

        # Extract summaries for inter-chunk recurrence
        total_A = cum_A[:, :, -1]         # (B, NC, H, 2, 2)
        final_local_h = local_h[:, :, -1] # (B, NC, H, 2, D)

        # === 4. Inter-Chunk Recurrence (Global) ===
        chunk_states = _inter_chunk_scan(total_A, final_local_h)

        # === 5. Correction (Broadcast) ===
        # True h_{t} = local_h_{t} + (cum_A_{t} @ chunk_state_{k})
        # cum_A: (B, NC, C, H, 2, 2), chunk_states: (B, NC, H, 2, D)
        correction = torch.einsum('bnchij,bnhjd->bnchid', cum_A, chunk_states)

        Y = local_h + correction

        # Reshape to flat sequence (B, L_pad, H, 2, D)
        Y = Y.reshape(B, -1, H, 2, D)

        # Unpad
        if pad_len > 0:
            Y = Y[:, :L]

        return Y, chunk_states
