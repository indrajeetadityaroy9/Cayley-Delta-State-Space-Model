"""State Space Duality (SSD) - Chunkwise Parallel KSSM.

This module implements the "Chunkwise Parallel Scan" algorithm (Mamba-2),
adapted for the 2x2 Cayley dynamics of KSSM.

Algorithm:
    1.  Split sequence into chunks of size Q (e.g., 64).
    2.  Intra-Chunk (Local): Compute local states assuming h_{-1}=0.
        Also compute cumulative transition operators (A_cum) for each chunk.
    3.  Inter-Chunk (Global): Recurrently update state h between chunks using
        the total chunk transition operators.
    4.  Combine: Correct local states using the propagated global chunk states.

This algorithm allows parallelization over chunks and utilizes Tensor Cores
for the dense matrix multiplications within chunks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Constants for chunking
CHUNK_SIZE = 64

@torch.compile
def _intra_chunk_scan(
    A_flat: Tensor,  # (B*NC, C, H, 2, 2)
    U_flat: Tensor,  # (B*NC, C, H, D, 2)
):
    """
    Matmul-based intra-chunk scan via quadratic form (Mamba-2 SSD dual).

    Instead of sequential recurrence h[t] = A[t] @ h[t-1] + U[t],
    materializes the causal transition kernel L[t,s] and computes
    all local states in parallel via a single einsum on Tensor Cores:

        local_h[t] = sum_{s<=t} L[t,s] @ U[s]

    where L[t,s] = A[t] @ ... @ A[s+1] (identity for t=s).
    """
    batch_chunks, C, H, D, _ = U_flat.shape
    device = U_flat.device
    dtype = U_flat.dtype

    # Step 1: Sequential prefix products for cum_A (C steps of 2x2 matmul)
    cum_A = torch.empty(batch_chunks, C, H, 2, 2, device=device, dtype=dtype)
    curr_A = torch.eye(2, device=device, dtype=dtype).view(1, 1, 2, 2).expand(batch_chunks, H, 2, 2).contiguous()
    for t in range(C):
        curr_A = torch.matmul(A_flat[:, t], curr_A)
        cum_A[:, t] = curr_A

    # Step 2: Build causal transition kernel L[t,s] = cum_A[t] @ cum_A[s]^{-1}
    # For 2x2 Cayley [[a,b],[-b,a]]: inv = [[a,-b],[b,a]] / (a²+b²)
    
    # === FIX: Compute inverse in FP32 to avoid blowups when det is small ===
    cum_A_fp32 = cum_A.float()
    cum_A_inv = cum_A_fp32.clone()
    cum_A_inv[..., 0, 1] = -cum_A_fp32[..., 0, 1]
    cum_A_inv[..., 1, 0] = -cum_A_fp32[..., 1, 0]
    det = cum_A_fp32[..., 0, 0].pow(2) + cum_A_fp32[..., 0, 1].pow(2)
    cum_A_inv = cum_A_inv / (det.unsqueeze(-1).unsqueeze(-1) + 1e-8)

    # L[t,s] = cum_A[t] @ cum_A_inv[s]
    # Compute L in FP32 for precision
    L = torch.einsum("bthij,bshjk->btshik", cum_A_fp32, cum_A_inv)  # (B*NC, C, C, H, 2, 2)

    # Causal mask: L[t,s] = 0 for s > t
    causal_mask = torch.tril(torch.ones(C, C, device=device, dtype=torch.float32))
    L = L * causal_mask.view(C, C, 1, 1, 1)

    # Cast back to input dtype for the main data contraction to save memory
    L = L.to(dtype)

    # Step 3: Parallel state computation via single einsum (Tensor Cores)
    local_h = torch.einsum("btshij,bshdj->bthdi", L, U_flat)

    return local_h, cum_A

@torch.compile
def _inter_chunk_scan(
    total_A: Tensor,      # (B, NC, H, 2, 2)
    final_local_h: Tensor, # (B, NC, H, D, 2)
):
    """
    Sequential recurrence across chunks.
    """
    B, n_chunks, H, D, _ = final_local_h.shape
    device = total_A.device
    dtype = total_A.dtype
    
    chunk_states = torch.empty(B, n_chunks, H, D, 2, device=device, dtype=dtype)
    
    # === FIX: Use FP32 for state accumulation to prevent swamping ===
    state = torch.zeros(B, H, D, 2, device=device, dtype=torch.float32)
    
    # Cast inputs to FP32 for the loop
    total_A_fp32 = total_A.float()
    final_local_h_fp32 = final_local_h.float()

    for k in range(n_chunks):
        chunk_states[:, k] = state.to(dtype)

        # Update for next chunk in FP32
        # h_{k+1_in} = total_A_k @ h_{k_in} + final_local_h_k
        state = torch.einsum("bhij,bhdj->bhdi", total_A_fp32[:, k], state) + final_local_h_fp32[:, k]

    return chunk_states

class SSDChunkwiseScan(nn.Module):
    """Chunkwise Parallel Scan for KSSM."""
    
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
    ) -> Tensor:
        """
        Args:
            alpha, omega, dt: (B, L, H)
            K: (B, L, H, D, 2) - reshaped per head
            V: (B, L, H, D, 1) - reshaped per head
        """
        B, L, H, D, _ = K.shape # K is (B, L, H, D, 2) based on caller
        
        # 1. Discretize Dynamics (Parallel)
        # Compute A_bar for every step: (B, L, H, 2, 2)
        # A = [[-alpha, omega], [-omega, -alpha]]
        tau = dt * 0.5
        tau_alpha = tau * alpha
        tau_omega = tau * omega
        denom = (1.0 + tau_alpha).pow(2) + tau_omega.pow(2) + 1e-6
        inv_det = 1.0 / denom
        
        # Elements of A_bar = (I - tau*A)^{-1}(I + tau*A)
        # a11 = ((1+tau*a)*(1-tau*a) - tau*w^2) / det
        a11 = ((1.0 + tau_alpha) * (1.0 - tau_alpha) - tau_omega.pow(2)) * inv_det
        a12 = (2.0 * tau_omega) * inv_det
        
        # A_bar: (B, L, H, 2, 2)
        A_bar = torch.stack([
            torch.stack([a11, a12], dim=-1),
            torch.stack([-a12, a11], dim=-1)
        ], dim=-2)
        
        # 2. Input Injection (Parallel)
        # U = (K * V) * dt
        # K: (B, L, H, D, 2), V: (B, L, H, D, 1)
        # U: (B, L, H, D, 2)
        U = (K * V) * dt.unsqueeze(-1).unsqueeze(-1)
        
        # 3. Chunk Preparation
        n_chunks = math.ceil(L / self.chunk_size)
        pad_len = n_chunks * self.chunk_size - L
        
        if pad_len > 0:
            # Pad L dimension
            A_bar = F.pad(A_bar, (0,0, 0,0, 0,0, 0,pad_len))
            U = F.pad(U, (0,0, 0,0, 0,0, 0,pad_len))
            
        # Reshape to (B, NC, C, H, ...)
        A_chunk = A_bar.view(B, n_chunks, self.chunk_size, H, 2, 2)
        U_chunk = U.view(B, n_chunks, self.chunk_size, H, D, 2)
        
        # Flatten for batched intra-chunk scan
        A_flat = A_chunk.view(B * n_chunks, self.chunk_size, H, 2, 2)
        U_flat = U_chunk.view(B * n_chunks, self.chunk_size, H, D, 2)
        
        # === 4. Intra-Chunk Scan (Local) ===
        local_h, cum_A = _intra_chunk_scan(A_flat, U_flat)
        
        # Reshape back to (B, NC, ...)
        local_h = local_h.view(B, n_chunks, self.chunk_size, H, D, 2)
        cum_A = cum_A.view(B, n_chunks, self.chunk_size, H, 2, 2)
        
        # Extract summaries for inter-chunk recurrence
        # total_A is the last cumulative operator in each chunk
        total_A = cum_A[:, :, -1] # (B, NC, H, 2, 2)
        # final_local_h is the last local state in each chunk
        final_local_h = local_h[:, :, -1] # (B, NC, H, D, 2)
        
        # === 5. Inter-Chunk Recurrence (Global) ===
        chunk_states = _inter_chunk_scan(total_A, final_local_h)
        
        # === 6. Correction (Broadcast) ===
        # True h_{t} = local_h_{t} + (cum_A_{t} @ chunk_state_{k})
        # cum_A: (B, NC, C, H, 2, 2) — cumulative transition per position
        # chunk_states: (B, NC, H, D, 2) — state entering each chunk
        # Einsum broadcasts over C (chunk position) automatically.
        correction = torch.einsum("bnchij,bnhdj->bnchdi", cum_A, chunk_states)

        # Y_true = local + correction
        Y = local_h + correction
        
        # Reshape to flat sequence (B, L_pad, H, D, 2)
        Y = Y.reshape(B, -1, H, D, 2)
        
        # Unpad
        if pad_len > 0:
            Y = Y[:, :L]
            
        return Y
