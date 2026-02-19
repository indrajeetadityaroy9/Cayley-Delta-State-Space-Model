"""State Space Duality (SSD) - Chunkwise Parallel CDSSM with Delta-Rule.

This module implements the chunkwise parallel scan algorithm (Mamba-2 style),
adapted for the complex diagonal Cayley dynamics of CDSSM v2 with gated
delta-rule state updates and N-dimensional state.

Algorithm:
    1.  Split sequence into chunks of size Q (e.g., 64).
    2.  Intra-Chunk (Local): Sequential delta-rule scan assuming h_{-1}=0.
        State is an (N, D) matrix memory per head. Keys are D-dimensional,
        values are N-dimensional (complex diagonal state vectors).
        h[t] = rot(A_bar[t], h[t-1] - beta*(h@k)*k^T) + beta*(v*k^T)
        Also tracks cumulative base A_bar operators (cum_A) for each chunk.
    3.  Inter-Chunk (Global): Recurrently update state h between chunks using
        the cumulative base transition operators (complex diagonal).
    4.  Exact Correction: Propagate inter-chunk states through intra-chunk
        dynamics (rotation + erasure, no injection) for exact correction.
        Uses fused CUDA kernel.

All scan operations use fused CUDA kernels (csrc/kernels/) for performance.
"""

import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cdssm.ops import IntraChunkScanFn, InterChunkScanFn, ExactCorrectionFn


class SSDChunkwiseScan(nn.Module):
    """Chunkwise Parallel Scan for CDSSM with delta-rule state updates."""

    def __init__(self, chunk_size: int, state_dim: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.state_dim = state_dim

    def forward(
        self,
        A_bar: Tensor,
        K: Tensor,
        V: Tensor,
        beta: Tensor,
    ) -> Tensor:
        """
        Args:
            A_bar: (B, L, H, N) - complex diagonal transition (re/im interleaved)
            K: (B, L, H, D) - D-dimensional keys per head
            V: (B, L, H, N) - N-dim value vectors per head (already VP-scaled)
            beta: (B, L, H) - delta-rule update gate

        Returns:
            Y: (B, L, H, N, D) - evolved matrix states
        """
        B, L, H, D = K.shape
        N = self.state_dim

        # 1. Chunk Preparation
        n_chunks = math.ceil(L / self.chunk_size)
        pad_len = n_chunks * self.chunk_size - L

        if pad_len > 0:
            A_bar = F.pad(A_bar, (0, 0, 0, 0, 0, pad_len))
            K = F.pad(K, (0, 0, 0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, 0, 0, pad_len))
            beta = F.pad(beta, (0, 0, 0, pad_len))

        # Reshape to (B, NC, C, H, ...)
        C = self.chunk_size
        A_chunk = A_bar.view(B, n_chunks, C, H, N)
        K_chunk = K.view(B, n_chunks, C, H, D)
        V_chunk = V.view(B, n_chunks, C, H, N)
        beta_chunk = beta.view(B, n_chunks, C, H)

        # Flatten for batched intra-chunk scan
        A_flat = A_chunk.reshape(B * n_chunks, C, H, N)
        K_flat = K_chunk.reshape(B * n_chunks, C, H, D)
        V_flat = V_chunk.reshape(B * n_chunks, C, H, N)
        beta_flat = beta_chunk.reshape(B * n_chunks, C, H)

        # === 2. Intra-Chunk Scan (CUDA Kernel) ===
        local_h, cum_A = IntraChunkScanFn.apply(A_flat, K_flat, V_flat, beta_flat, N)

        # Reshape back to (B, NC, ...)
        local_h = local_h.view(B, n_chunks, C, H, N, D)
        cum_A = cum_A.view(B, n_chunks, C, H, N)

        # Extract summaries for inter-chunk recurrence
        total_A = cum_A[:, :, -1]         # (B, NC, H, N)
        final_local_h = local_h[:, :, -1] # (B, NC, H, N, D)

        # === 3. Inter-Chunk Recurrence (CUDA Kernel) ===
        chunk_states = InterChunkScanFn.apply(total_A, final_local_h, N)

        # === 4. Exact Correction (Fused CUDA Kernel) ===
        # Propagate chunk_states through intra-chunk dynamics (erasure + rotation)
        correction = ExactCorrectionFn.apply(
            A_chunk, K_chunk, beta_chunk, chunk_states, N
        )

        Y = local_h + correction

        # Reshape to flat sequence (B, L_pad, H, N, D)
        Y = Y.reshape(B, -1, H, N, D)

        # Unpad
        if pad_len > 0:
            Y = Y[:, :L]

        return Y
