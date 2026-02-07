"""Tests for SSD chunkwise scan correctness against naive sequential recurrence."""

import torch
import torch.nn.functional as F

from kssm.modules.ssd import SSDChunkwiseScan


def naive_recurrence(A_bar, U, initial_state=None):
    """Reference sequential implementation.

    Args:
        A_bar: (B, L, H, 2, 2) — per-step transition matrices
        U: (B, L, H, D, 2) — per-step input injections

    Returns:
        outputs: (B, L, H, D, 2) — state at every timestep
        final_state: (B, H, D, 2) — state after last timestep
    """
    B, L, H, D, _ = U.shape

    if initial_state is None:
        h = torch.zeros(B, H, D, 2, device=U.device, dtype=U.dtype)
    else:
        h = initial_state

    outputs = []
    for t in range(L):
        a_t = A_bar[:, t]  # (B, H, 2, 2)
        u_t = U[:, t]      # (B, H, D, 2)
        h = torch.einsum("bhij,bhdj->bhdi", a_t, h) + u_t
        outputs.append(h)

    return torch.stack(outputs, dim=1), h


def build_cayley_A_bar(alpha, omega, dt):
    """Construct A_bar via Cayley discretization from alpha, omega, dt.

    All inputs: (B, L, H). Returns: (B, L, H, 2, 2).
    """
    tau = dt * 0.5
    tau_alpha = tau * alpha
    tau_omega = tau * omega
    denom = (1.0 + tau_alpha).pow(2) + tau_omega.pow(2) + 1e-6
    inv_det = 1.0 / denom

    a11 = ((1.0 + tau_alpha) * (1.0 - tau_alpha) - tau_omega.pow(2)) * inv_det
    a12 = (2.0 * tau_omega) * inv_det

    A_bar = torch.stack([
        torch.stack([a11, a12], dim=-1),
        torch.stack([-a12, a11], dim=-1)
    ], dim=-2)

    return A_bar


def test_ssd_matches_naive():
    """Test that SSD chunkwise scan matches naive recurrence (zero initial state)."""
    torch.manual_seed(42)

    B, L, H, D = 2, 128, 4, 8
    d_inner = H * D

    alpha = F.softplus(torch.randn(B, L, H)) * 0.5
    omega = torch.randn(B, L, H) * 2.0
    dt = torch.ones(B, L, H) * 0.1

    K = torch.randn(B, L, H, D, 2) * 0.1
    V = torch.randn(B, L, H, D, 1) * 0.1

    # Reference: build A_bar, compute U, run naive recurrence
    A_bar = build_cayley_A_bar(alpha, omega, dt)
    U = (K * V) * dt.unsqueeze(-1).unsqueeze(-1)
    y_naive, final_naive = naive_recurrence(A_bar, U)

    # SSD chunkwise scan
    ssd = SSDChunkwiseScan(d_inner, H)
    y_ssd, final_ssd = ssd(alpha, omega, dt, K, V)

    diff_y = (y_naive - y_ssd).abs().max().item()
    diff_final = (final_naive - final_ssd).abs().max().item()

    print(f"Naive shape: {y_naive.shape}, SSD shape: {y_ssd.shape}")
    print(f"Max Y difference: {diff_y:.2e}")
    print(f"Max final state difference: {diff_final:.2e}")

    assert diff_y < 1e-4, f"Y mismatch: {diff_y}"
    assert diff_final < 1e-4, f"Final state mismatch: {diff_final}"
    print("PASSED: SSD matches naive recurrence")


def test_ssd_with_initial_state():
    """Test SSD with non-zero initial state."""
    torch.manual_seed(123)

    B, L, H, D = 1, 64, 2, 4
    d_inner = H * D

    alpha = F.softplus(torch.randn(B, L, H)) * 0.3
    omega = torch.randn(B, L, H)
    dt = torch.ones(B, L, H) * 0.2

    K = torch.randn(B, L, H, D, 2) * 0.1
    V = torch.randn(B, L, H, D, 1) * 0.1

    initial_state = torch.randn(B, H, D, 2) * 0.5

    A_bar = build_cayley_A_bar(alpha, omega, dt)
    U = (K * V) * dt.unsqueeze(-1).unsqueeze(-1)
    y_naive, final_naive = naive_recurrence(A_bar, U, initial_state)

    ssd = SSDChunkwiseScan(d_inner, H)
    y_ssd, final_ssd = ssd(alpha, omega, dt, K, V, initial_state=initial_state)

    diff_y = (y_naive - y_ssd).abs().max().item()
    diff_final = (final_naive - final_ssd).abs().max().item()

    print(f"Max Y difference (with init state): {diff_y:.2e}")
    print(f"Max final state difference: {diff_final:.2e}")

    assert diff_y < 1e-4, f"Y mismatch with initial state: {diff_y}"
    assert diff_final < 1e-4, f"Final state mismatch: {diff_final}"
    print("PASSED: SSD with initial state matches naive")


def test_ssd_padding():
    """Test SSD when sequence length is not a multiple of chunk size (64)."""
    torch.manual_seed(99)

    B, L, H, D = 1, 100, 2, 4  # L=100, not divisible by 64
    d_inner = H * D

    alpha = F.softplus(torch.randn(B, L, H)) * 0.3
    omega = torch.randn(B, L, H)
    dt = torch.ones(B, L, H) * 0.15

    K = torch.randn(B, L, H, D, 2) * 0.1
    V = torch.randn(B, L, H, D, 1) * 0.1

    A_bar = build_cayley_A_bar(alpha, omega, dt)
    U = (K * V) * dt.unsqueeze(-1).unsqueeze(-1)
    y_naive, final_naive = naive_recurrence(A_bar, U)

    ssd = SSDChunkwiseScan(d_inner, H)
    y_ssd, final_ssd = ssd(alpha, omega, dt, K, V)

    # Output should be unpadded back to L=100
    assert y_ssd.shape[1] == L, f"Expected L={L}, got {y_ssd.shape[1]}"

    diff_y = (y_naive - y_ssd).abs().max().item()
    print(f"Max Y difference (padded L={L}): {diff_y:.2e}")
    assert diff_y < 1e-4, f"Y mismatch with padding: {diff_y}"
    print("PASSED: SSD padding correctness")


if __name__ == "__main__":
    test_ssd_matches_naive()
    test_ssd_with_initial_state()
    test_ssd_padding()
