"""Numerical correctness tests: Python reference vs CUDA kernels.

For each of the 4 CUDA kernels, we implement a pure-Python/PyTorch reference
and compare outputs against the CUDA kernel using torch.allclose with BF16
tolerances.

Test shapes: B=2, L=256, H=6, D=64, C=64 (chunk size).
"""

import math
import pytest
import torch
import torch.nn.functional as F


# ============================================================================
# Python Reference Implementations
# ============================================================================

def ref_cayley_discretize(alpha, omega, dt, eps=1e-6):
    """Python reference for Cayley discretization (matches cayley_math.cuh)."""
    alpha = alpha.clamp(min=0.0)
    tau = dt * 0.5
    tau_omega = tau * omega
    one_plus_tau_alpha = 1.0 + tau * alpha
    one_minus_tau_alpha = 1.0 - tau * alpha

    det_M = one_plus_tau_alpha ** 2 + tau_omega ** 2
    inv_det = 1.0 / (det_M + eps)

    # M^{-1}
    m11 = one_plus_tau_alpha * inv_det
    m12 = tau_omega * inv_det
    m21 = -tau_omega * inv_det
    m22 = m11

    # A_bar = M^{-1} @ N, N = [[1-τα, τω], [-τω, 1-τα]]
    a11 = m11 * one_minus_tau_alpha + m12 * (-tau_omega)
    a12 = m11 * tau_omega + m12 * one_minus_tau_alpha
    a21 = m21 * one_minus_tau_alpha + m22 * (-tau_omega)
    a22 = m21 * tau_omega + m22 * one_minus_tau_alpha

    return a11, a12, a21, a22


def ref_cayley_vp(alpha, omega, dt, r_gate, gating_c, eps=1e-6):
    """Python reference for fused Cayley + VP scale (matches cayley_vp.cu)."""
    a11, a12, a21, a22 = ref_cayley_discretize(alpha, omega, dt, eps)

    alpha_clamped = alpha.clamp(min=0.0)
    tau = dt * 0.5
    tau_a = tau * alpha_clamped
    tau_w = tau * omega

    numer = (1.0 - tau_a) ** 2 + tau_w ** 2
    denom = (1.0 + tau_a) ** 2 + tau_w ** 2
    eig_sq = numer / (denom + eps)

    eig_sq_clamped = eig_sq.clamp(min=1e-8)

    if r_gate is not None and r_gate.numel() > 0:
        exponent = (gating_c * r_gate - 1.0) / 2.0
        scale = eig_sq_clamped.pow(exponent)
        a11 = a11 * scale
        a12 = a12 * scale
        a21 = a21 * scale
        a22 = a22 * scale
        effective_eig_sq = eig_sq_clamped.pow(gating_c * r_gate)
    else:
        effective_eig_sq = eig_sq

    vp = (1.0 - effective_eig_sq).clamp(min=eps).sqrt().clamp(max=1.0)

    # Stack A_bar: (B, L, H, 2, 2)
    A_bar = torch.stack([
        torch.stack([a11, a12], dim=-1),
        torch.stack([a21, a22], dim=-1),
    ], dim=-2)

    return A_bar, vp


def ref_adaptive_dt(alpha, omega, log_dt_scale, omega_thresh, delta, smoothness, eps):
    """Python reference for adaptive timestep (matches adaptive_dt.cu)."""
    dt_scale = F.softplus(log_dt_scale)  # (H,)
    char_freq = alpha + omega.abs() + eps
    dt_raw = dt_scale / char_freq

    dt_max = (2.0 - delta) / (alpha + eps)

    omega_abs = omega.abs()
    blend = torch.sigmoid((omega_thresh - omega_abs) / smoothness)

    dt_capped = torch.min(dt_raw, dt_max)
    dt_val = blend * dt_capped + (1.0 - blend) * dt_raw

    return dt_val


def ref_intra_chunk_scan(A_flat, K_flat, V_flat, beta_flat):
    """Python reference for intra-chunk delta-rule scan (matches intra_chunk_scan.cu).

    Args:
        A_flat: (BNC, C, H, 2, 2)
        K_flat: (BNC, C, H, D)
        V_flat: (BNC, C, H, 2)
        beta_flat: (BNC, C, H)

    Returns:
        local_h: (BNC, C, H, 2, D)
        cum_A: (BNC, C, H, 2, 2)
    """
    BNC, C, H, D = K_flat.shape
    device = K_flat.device
    dtype = K_flat.dtype

    local_h = torch.zeros(BNC, C, H, 2, D, device=device, dtype=torch.float32)
    cum_A = torch.zeros(BNC, C, H, 2, 2, device=device, dtype=torch.float32)

    # Convert to float for computation
    A = A_flat.float()
    K = K_flat.float()
    V = V_flat.float()
    beta = beta_flat.float()

    for n in range(BNC):
        for h in range(H):
            # State: (2, D)
            state = torch.zeros(2, D, device=device, dtype=torch.float32)
            # Cumulative A: (2, 2) identity
            cA = torch.eye(2, device=device, dtype=torch.float32)

            for t in range(C):
                a = A[n, t, h]  # (2, 2)
                k = K[n, t, h]  # (D,)
                v = V[n, t, h]  # (2,)
                b = beta[n, t, h]  # scalar

                # 1. Retrieval: kTh = state @ k -> (2,)
                kTh = state @ k

                # 2. Erasure: state -= beta * outer(kTh, k)
                state = state - b * torch.outer(kTh, k)

                # 3. Rotation: state = A @ state
                state = a @ state

                # 4. Injection: state += beta * outer(v, k)
                state = state + b * torch.outer(v, k)

                local_h[n, t, h] = state

                # Update cumulative A: cA = A[t] @ cA
                cA = a @ cA
                cum_A[n, t, h] = cA

    return local_h.to(dtype), cum_A.to(dtype)


def ref_inter_chunk_scan(total_A, final_local_h):
    """Python reference for inter-chunk scan (matches inter_chunk_scan.cu).

    Args:
        total_A: (B, NC, H, 2, 2)
        final_local_h: (B, NC, H, 2, D)

    Returns:
        chunk_states: (B, NC, H, 2, D)
    """
    B, NC, H = total_A.shape[:3]
    D = final_local_h.shape[4]
    device = total_A.device
    dtype = total_A.dtype

    chunk_states = torch.zeros(B, NC, H, 2, D, device=device, dtype=torch.float32)

    tA = total_A.float()
    flh = final_local_h.float()

    for b in range(B):
        for h in range(H):
            state = torch.zeros(2, D, device=device, dtype=torch.float32)
            for k in range(NC):
                # Store BEFORE update
                chunk_states[b, k, h] = state
                # Update
                state = tA[b, k, h] @ state + flh[b, k, h]

    return chunk_states.to(dtype)


# ============================================================================
# Tests
# ============================================================================

@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "CUDA required"
    return torch.device("cuda")


class TestCayleyVP:
    """Test cayley_vp_cuda against Python reference."""

    def test_forward_with_gate(self, device):
        from kssm.ops import cayley_vp_cuda

        B, L, H = 2, 256, 6
        torch.manual_seed(42)
        alpha = torch.rand(B, L, H, device=device, dtype=torch.bfloat16) * 2.0
        omega = torch.randn(B, L, H, device=device, dtype=torch.bfloat16)
        dt = torch.rand(B, L, H, device=device, dtype=torch.bfloat16) * 0.5 + 0.01
        r_gate = torch.sigmoid(torch.randn(B, L, H, device=device, dtype=torch.bfloat16))
        gating_c = math.log(8192)

        # CUDA
        A_bar_cuda, vp_cuda = cayley_vp_cuda(alpha, omega, dt, r_gate, gating_c)

        # Python reference (use float for accuracy, then compare in bf16)
        A_bar_ref, vp_ref = ref_cayley_vp(
            alpha.float(), omega.float(), dt.float(), r_gate.float(), gating_c
        )
        A_bar_ref = A_bar_ref.bfloat16()
        vp_ref = vp_ref.bfloat16()

        assert torch.allclose(A_bar_cuda, A_bar_ref, atol=1e-2, rtol=1e-2), \
            f"A_bar mismatch: max diff = {(A_bar_cuda - A_bar_ref).abs().max():.6f}"
        assert torch.allclose(vp_cuda, vp_ref, atol=1e-2, rtol=1e-2), \
            f"vp_scale mismatch: max diff = {(vp_cuda - vp_ref).abs().max():.6f}"

    def test_forward_without_gate(self, device):
        from kssm.ops import cayley_vp_cuda

        B, L, H = 2, 128, 6
        torch.manual_seed(123)
        alpha = torch.rand(B, L, H, device=device, dtype=torch.bfloat16) * 2.0
        omega = torch.randn(B, L, H, device=device, dtype=torch.bfloat16)
        dt = torch.rand(B, L, H, device=device, dtype=torch.bfloat16) * 0.5 + 0.01
        r_gate = torch.empty(0, device=device, dtype=torch.bfloat16)
        gating_c = math.log(8192)

        A_bar_cuda, vp_cuda = cayley_vp_cuda(alpha, omega, dt, r_gate, gating_c)

        A_bar_ref, vp_ref = ref_cayley_vp(
            alpha.float(), omega.float(), dt.float(), None, gating_c
        )
        A_bar_ref = A_bar_ref.bfloat16()
        vp_ref = vp_ref.bfloat16()

        assert torch.allclose(A_bar_cuda, A_bar_ref, atol=1e-2, rtol=1e-2)
        assert torch.allclose(vp_cuda, vp_ref, atol=1e-2, rtol=1e-2)


class TestAdaptiveDt:
    """Test adaptive_dt_cuda against Python reference."""

    def test_forward(self, device):
        from kssm.ops import adaptive_dt_cuda

        B, L, H = 2, 256, 6
        torch.manual_seed(42)
        alpha = F.softplus(torch.randn(B, L, H, device=device, dtype=torch.bfloat16))
        omega = torch.randn(B, L, H, device=device, dtype=torch.bfloat16)
        log_dt_scale = torch.randn(H, device=device, dtype=torch.float32)

        # Derive safety constants (matching components.py bf16_safety_constants)
        eps_bf16 = torch.finfo(torch.bfloat16).eps
        omega_thresh = math.sqrt(eps_bf16)
        delta = 16.0 * eps_bf16
        smoothness = omega_thresh / 5.0
        eps = eps_bf16 * 100

        dt_cuda = adaptive_dt_cuda(
            alpha, omega, log_dt_scale,
            omega_thresh, delta, smoothness, eps,
        )

        dt_ref = ref_adaptive_dt(
            alpha.float(), omega.float(), log_dt_scale,
            omega_thresh, delta, smoothness, eps,
        ).bfloat16()

        assert torch.allclose(dt_cuda, dt_ref, atol=1e-2, rtol=1e-2), \
            f"adaptive_dt mismatch: max diff = {(dt_cuda.float() - dt_ref.float()).abs().max():.6f}"


class TestIntraChunkScan:
    """Test intra_chunk_scan_cuda against Python reference."""

    def test_forward(self, device):
        from kssm.ops import intra_chunk_scan_cuda

        # Use smaller shapes to keep the Python reference tractable
        BNC, C, H, D = 4, 16, 4, 32
        torch.manual_seed(42)

        # Generate A_bar matrices with proper Cayley structure (orthogonal-ish)
        theta = torch.randn(BNC, C, H, device=device) * 0.5
        decay = torch.rand(BNC, C, H, device=device) * 0.1 + 0.9
        cos_t = torch.cos(theta) * decay
        sin_t = torch.sin(theta) * decay

        A_flat = torch.zeros(BNC, C, H, 2, 2, device=device, dtype=torch.bfloat16)
        A_flat[..., 0, 0] = cos_t.bfloat16()
        A_flat[..., 0, 1] = sin_t.bfloat16()
        A_flat[..., 1, 0] = (-sin_t).bfloat16()
        A_flat[..., 1, 1] = cos_t.bfloat16()

        K_flat = torch.randn(BNC, C, H, D, device=device, dtype=torch.bfloat16) * 0.1
        # Normalize keys (matching kssm_block.py)
        K_flat = F.normalize(K_flat.float(), dim=-1).bfloat16()
        V_flat = torch.randn(BNC, C, H, 2, device=device, dtype=torch.bfloat16) * 0.1
        beta_flat = torch.sigmoid(torch.randn(BNC, C, H, device=device, dtype=torch.bfloat16))

        # CUDA
        local_h_cuda, cum_A_cuda = intra_chunk_scan_cuda(A_flat, K_flat, V_flat, beta_flat)

        # Python reference
        local_h_ref, cum_A_ref = ref_intra_chunk_scan(A_flat, K_flat, V_flat, beta_flat)

        # BF16 tolerances are larger due to sequential accumulation
        assert torch.allclose(local_h_cuda.float(), local_h_ref.float(), atol=5e-2, rtol=5e-2), \
            f"local_h mismatch: max diff = {(local_h_cuda.float() - local_h_ref.float()).abs().max():.6f}"
        assert torch.allclose(cum_A_cuda.float(), cum_A_ref.float(), atol=5e-2, rtol=5e-2), \
            f"cum_A mismatch: max diff = {(cum_A_cuda.float() - cum_A_ref.float()).abs().max():.6f}"


class TestInterChunkScan:
    """Test inter_chunk_scan_cuda against Python reference."""

    def test_forward(self, device):
        from kssm.ops import inter_chunk_scan_cuda

        B, NC, H, D = 2, 8, 6, 64
        torch.manual_seed(42)

        # Generate total_A as near-orthogonal 2x2 (Cayley-like)
        theta = torch.randn(B, NC, H, device=device) * 0.3
        decay = torch.rand(B, NC, H, device=device) * 0.1 + 0.85
        cos_t = torch.cos(theta) * decay
        sin_t = torch.sin(theta) * decay

        total_A = torch.zeros(B, NC, H, 2, 2, device=device, dtype=torch.bfloat16)
        total_A[..., 0, 0] = cos_t.bfloat16()
        total_A[..., 0, 1] = sin_t.bfloat16()
        total_A[..., 1, 0] = (-sin_t).bfloat16()
        total_A[..., 1, 1] = cos_t.bfloat16()

        final_local_h = torch.randn(B, NC, H, 2, D, device=device, dtype=torch.bfloat16) * 0.1

        # CUDA
        cs_cuda = inter_chunk_scan_cuda(total_A, final_local_h)

        # Python reference
        cs_ref = ref_inter_chunk_scan(total_A, final_local_h)

        assert torch.allclose(cs_cuda.float(), cs_ref.float(), atol=5e-2, rtol=5e-2), \
            f"chunk_states mismatch: max diff = {(cs_cuda.float() - cs_ref.float()).abs().max():.6f}"


class TestEndToEndScan:
    """Test the full SSD chunkwise scan pipeline."""

    def test_ssd_pipeline(self, device):
        """Test that intra + inter + correction matches a flat sequential scan."""
        from kssm.ops import intra_chunk_scan_cuda, inter_chunk_scan_cuda

        B, L, H, D = 1, 128, 2, 32
        C = 64
        torch.manual_seed(42)

        # Generate sequence-level inputs
        theta = torch.randn(B, L, H, device=device) * 0.3
        decay = torch.rand(B, L, H, device=device) * 0.1 + 0.9
        cos_t = torch.cos(theta) * decay
        sin_t = torch.sin(theta) * decay

        A = torch.zeros(B, L, H, 2, 2, device=device, dtype=torch.bfloat16)
        A[..., 0, 0] = cos_t.bfloat16()
        A[..., 0, 1] = sin_t.bfloat16()
        A[..., 1, 0] = (-sin_t).bfloat16()
        A[..., 1, 1] = cos_t.bfloat16()

        K = F.normalize(torch.randn(B, L, H, D, device=device).float(), dim=-1).bfloat16()
        V = torch.randn(B, L, H, 2, device=device, dtype=torch.bfloat16) * 0.1
        beta = torch.sigmoid(torch.randn(B, L, H, device=device, dtype=torch.bfloat16))

        # --- Flat sequential reference ---
        local_h_flat, _ = ref_intra_chunk_scan(
            A.view(B, L, H, 2, 2), K, V, beta
        )
        # For the flat case with 1 chunk of size L, there's no inter-chunk correction needed
        # Instead, compute the full sequential scan
        ref_Y = torch.zeros(B, L, H, 2, D, device=device, dtype=torch.float32)
        A_f = A.float()
        K_f = K.float()
        V_f = V.float()
        beta_f = beta.float()

        for b_idx in range(B):
            for h_idx in range(H):
                state = torch.zeros(2, D, device=device, dtype=torch.float32)
                for t in range(L):
                    a = A_f[b_idx, t, h_idx]
                    k = K_f[b_idx, t, h_idx]
                    v = V_f[b_idx, t, h_idx]
                    bt = beta_f[b_idx, t, h_idx]
                    kTh = state @ k
                    state = state - bt * torch.outer(kTh, k)
                    state = a @ state
                    state = state + bt * torch.outer(v, k)
                    ref_Y[b_idx, t, h_idx] = state

        # --- Chunkwise pipeline ---
        n_chunks = L // C
        A_chunk = A.view(B, n_chunks, C, H, 2, 2)
        K_chunk = K.view(B, n_chunks, C, H, D)
        V_chunk = V.view(B, n_chunks, C, H, 2)
        beta_chunk = beta.view(B, n_chunks, C, H)

        A_flat = A_chunk.reshape(B * n_chunks, C, H, 2, 2)
        K_flat = K_chunk.reshape(B * n_chunks, C, H, D)
        V_flat = V_chunk.reshape(B * n_chunks, C, H, 2)
        beta_flat = beta_chunk.reshape(B * n_chunks, C, H)

        local_h, cum_A = intra_chunk_scan_cuda(A_flat, K_flat, V_flat, beta_flat)

        local_h = local_h.view(B, n_chunks, C, H, 2, D)
        cum_A = cum_A.view(B, n_chunks, C, H, 2, 2)

        total_A = cum_A[:, :, -1]
        final_local_h = local_h[:, :, -1]

        chunk_states = inter_chunk_scan_cuda(total_A, final_local_h)

        # Correction
        correction = torch.einsum('bnchij,bnhjd->bnchid', cum_A, chunk_states)
        Y = local_h + correction
        Y = Y.reshape(B, L, H, 2, D)

        # Compare (BF16 accumulation error grows over L steps)
        assert torch.allclose(Y.float(), ref_Y, atol=0.15, rtol=0.15), \
            f"SSD pipeline mismatch: max diff = {(Y.float() - ref_Y).abs().max():.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
