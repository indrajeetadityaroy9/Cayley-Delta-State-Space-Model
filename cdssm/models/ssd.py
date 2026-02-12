"""State Space Duality (SSD) - Chunkwise Parallel CDSSM with Delta-Rule.

This module implements the chunkwise parallel scan algorithm (Mamba-2 style),
adapted for the 2x2 Cayley dynamics of CDSSM with gated delta-rule state updates.

Algorithm:
    1.  Split sequence into chunks of size Q (e.g., 64).
    2.  Intra-Chunk (Local): Sequential delta-rule scan assuming h_{-1}=0.
        State is a (2, D) matrix memory per head. Keys are D-dimensional,
        values are 2-dimensional (Hamiltonian state vectors).
        h[t] = A_bar[t] @ (h[t-1] - beta*(h@k)*k^T) + beta*(v*k^T)
        Also tracks cumulative base A_bar operators (cum_A) for each chunk.
    3.  Inter-Chunk (Global): Recurrently update state h between chunks using
        the cumulative base transition operators.
    4.  Combine: Correct local states using the propagated global chunk states.

The delta-rule erasure (step 2) enables selective forgetting of specific
associations without uniform decay. Inter-chunk correction uses base A_bar
products (without erasure), which is an acceptable approximation since most
associations form and are erased within a single chunk of 64 steps.

All scan operations use fused CUDA kernels (csrc/kernels/) for performance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cdssm.ops import intra_chunk_scan_cuda, inter_chunk_scan_cuda

# Constants for chunking
CHUNK_SIZE = 64  # = head_dim, aligned to H100 Tensor Core width


class SSDChunkwiseScan(nn.Module):
    """Chunkwise Parallel Scan for CDSSM with delta-rule state updates."""

    def __init__(self, d_inner: int, n_heads: int, chunk_size: int = CHUNK_SIZE):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.head_dim = d_inner // n_heads

    def forward(
        self,
        A_bar: Tensor,
        K: Tensor,
        V: Tensor,
        beta: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            A_bar: (B, L, H, 2, 2) - pre-computed discretized transition matrices
            K: (B, L, H, D) - D-dimensional keys per head
            V: (B, L, H, 2) - 2D Hamiltonian value vectors per head (already VP-scaled)
            beta: (B, L, H) - delta-rule update gate

        Returns:
            Y: (B, L, H, 2, D) - evolved matrix states
            chunk_states: (B, NC, H, 2, D) - inter-chunk states for feedback
        """
        B, L, H, D = K.shape

        # 1. Chunk Preparation
        n_chunks = math.ceil(L / self.chunk_size)
        pad_len = n_chunks * self.chunk_size - L

        if pad_len > 0:
            A_bar = F.pad(A_bar, (0, 0, 0, 0, 0, 0, 0, pad_len))
            K = F.pad(K, (0, 0, 0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, 0, 0, pad_len))
            beta = F.pad(beta, (0, 0, 0, pad_len))

        # Reshape to (B, NC, C, H, ...)
        C = self.chunk_size
        A_chunk = A_bar.view(B, n_chunks, C, H, 2, 2)
        K_chunk = K.view(B, n_chunks, C, H, D)
        V_chunk = V.view(B, n_chunks, C, H, 2)
        beta_chunk = beta.view(B, n_chunks, C, H)

        # Flatten for batched intra-chunk scan
        A_flat = A_chunk.reshape(B * n_chunks, C, H, 2, 2)
        K_flat = K_chunk.reshape(B * n_chunks, C, H, D)
        V_flat = V_chunk.reshape(B * n_chunks, C, H, 2)
        beta_flat = beta_chunk.reshape(B * n_chunks, C, H)

        # === 2. Intra-Chunk Scan (CUDA Kernel) ===
        local_h, cum_A = intra_chunk_scan_cuda(A_flat, K_flat, V_flat, beta_flat)

        # Reshape back to (B, NC, ...)
        local_h = local_h.view(B, n_chunks, C, H, 2, D)
        cum_A = cum_A.view(B, n_chunks, C, H, 2, 2)

        # Extract summaries for inter-chunk recurrence
        total_A = cum_A[:, :, -1]         # (B, NC, H, 2, 2)
        final_local_h = local_h[:, :, -1] # (B, NC, H, 2, D)

        # === 3. Inter-Chunk Recurrence (CUDA Kernel) ===
        chunk_states = inter_chunk_scan_cuda(total_A, final_local_h)

        # === 4. Correction (Broadcast) ===
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
