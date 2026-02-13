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
    4.  Exact Correction: Propagate inter-chunk states through intra-chunk
        dynamics (rotation + erasure, no injection) for exact correction.
        This is possible because the delta-rule dynamics are linear in the
        initial state h_init.

All scan operations use fused CUDA kernels (csrc/kernels/) for performance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cdssm.ops import intra_chunk_scan_cuda, inter_chunk_scan_cuda


class SSDChunkwiseScan(nn.Module):
    """Chunkwise Parallel Scan for CDSSM with delta-rule state updates."""

    def __init__(self, d_inner: int, n_heads: int, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size

    @staticmethod
    def _exact_correction_scan(
        A_chunk: Tensor,
        K_chunk: Tensor,
        beta_chunk: Tensor,
        chunk_states: Tensor,
    ) -> Tensor:
        """Propagate chunk initial states through dynamics with erasure (no injection).

        Since the delta-rule dynamics h[t] = A[t] @ (h[t-1] - beta*outer(h@k, k)) + beta*outer(v,k)
        are LINEAR in h (for fixed A, k, beta), superposition holds:
            h(h_init) = h(0) + Phi(t) @ h_init
        where Phi is the state transition matrix including both rotation AND erasure.

        This computes Phi(t) @ h_init exactly by running the dynamics with V=0,
        replacing the approximate correction einsum(cum_A, chunk_states) which
        ignores delta-rule erasure terms.

        Args:
            A_chunk:      (B, NC, C, H, 2, 2) - transition matrices per position
            K_chunk:      (B, NC, C, H, D)     - normalized keys per position
            beta_chunk:   (B, NC, C, H)        - delta-rule gates per position
            chunk_states: (B, NC, H, 2, D)     - initial state for each chunk

        Returns:
            corrections:  (B, NC, C, H, 2, D)  - exact correction at each position
        """
        B, NC, C, H = beta_chunk.shape

        # Work in float32 for numerical accuracy
        h = chunk_states.float().clone()  # (B, NC, H, 2, D)

        corrections = []
        for t in range(C):  # Only 64 iterations, vectorized over B, NC, H, D
            a = A_chunk[:, :, t].float()       # (B, NC, H, 2, 2)
            k = K_chunk[:, :, t].float()       # (B, NC, H, D)
            b = beta_chunk[:, :, t].float()    # (B, NC, H)

            # 1. Erasure: h -= beta * outer(h @ k, k)
            kTh = torch.einsum('bnhsd,bnhd->bnhs', h, k)  # (B, NC, H, 2)
            h = h - b[..., None, None] * torch.einsum('bnhs,bnhd->bnhsd', kTh, k)

            # 2. Rotation: h = A @ h
            h = torch.einsum('bnhij,bnhjd->bnhid', a, h)

            corrections.append(h.unsqueeze(2))  # (B, NC, 1, H, 2, D)

        return torch.cat(corrections, dim=2).to(A_chunk.dtype)  # (B, NC, C, H, 2, D)

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

        # === 4. Exact Correction ===
        # Propagate chunk_states through intra-chunk dynamics (erasure + rotation)
        # This is exact because the delta-rule dynamics are linear in h_init
        correction = self._exact_correction_scan(
            A_chunk, K_chunk, beta_chunk, chunk_states
        )

        Y = local_h + correction

        # Reshape to flat sequence (B, L_pad, H, 2, D)
        Y = Y.reshape(B, -1, H, 2, D)

        # Unpad
        if pad_len > 0:
            Y = Y[:, :L]

        return Y, chunk_states
