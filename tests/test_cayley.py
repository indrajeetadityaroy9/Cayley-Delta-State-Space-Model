"""Tests for Cayley transform discretization."""

import pytest
import torch

from kssm.kernels.cayley_fused import cayley_fused, cayley_fused_pytorch
from kssm.ops.cayley_op import cayley_transform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestCayleyTransform:
    """Test Cayley transform kernel."""

    @pytest.mark.parametrize("batch,seq_len,d_inner", [
        (1, 16, 32),
        (4, 64, 64),
        (8, 128, 128),
    ])
    def test_triton_matches_pytorch(self, batch, seq_len, d_inner):
        """Verify Triton kernel matches PyTorch reference."""
        torch.manual_seed(42)

        alpha = torch.rand(batch, seq_len, d_inner, device='cuda') * 0.5
        omega = torch.randn(batch, seq_len, d_inner, device='cuda')
        dt = torch.rand(batch, seq_len, d_inner, device='cuda') * 0.1 + 0.001
        B = torch.randn(batch, seq_len, d_inner, 2, device='cuda')
        x = torch.randn(batch, seq_len, d_inner, device='cuda')

        # PyTorch reference
        A_bar_ref, u_bar_ref = cayley_fused_pytorch(alpha, omega, dt, B, x)

        # Triton
        A_bar_tri, u_bar_tri = cayley_fused(
            alpha.bfloat16(), omega.bfloat16(), dt.bfloat16(),
            B.bfloat16(), x.bfloat16()
        )

        # Compare
        torch.testing.assert_close(
            A_bar_tri.float(), A_bar_ref.float(),
            rtol=5e-2, atol=5e-2,
        )
        torch.testing.assert_close(
            u_bar_tri.float(), u_bar_ref.float(),
            rtol=5e-2, atol=5e-2,
        )

    def test_determinant_stability(self):
        """Test that epsilon prevents division by zero."""
        batch, seq_len, d_inner = 2, 16, 32

        # Very small alpha and omega -> det_M close to 1, should be stable
        alpha = torch.zeros(batch, seq_len, d_inner, device='cuda')
        omega = torch.zeros(batch, seq_len, d_inner, device='cuda')
        dt = torch.ones(batch, seq_len, d_inner, device='cuda') * 0.01
        B = torch.ones(batch, seq_len, d_inner, 2, device='cuda')
        x = torch.ones(batch, seq_len, d_inner, device='cuda')

        A_bar, u_bar = cayley_fused(
            alpha.bfloat16(), omega.bfloat16(), dt.bfloat16(),
            B.bfloat16(), x.bfloat16()
        )

        # Should not have NaN or Inf
        assert torch.isfinite(A_bar).all(), "A_bar has non-finite values"
        assert torch.isfinite(u_bar).all(), "u_bar has non-finite values"

    def test_rotation_structure(self):
        """Test that A_bar has rotation structure for zero damping."""
        batch, seq_len, d_inner = 2, 16, 32

        # Zero damping -> pure rotation
        alpha = torch.zeros(batch, seq_len, d_inner, device='cuda')
        omega = torch.ones(batch, seq_len, d_inner, device='cuda')  # Unit frequency
        dt = torch.ones(batch, seq_len, d_inner, device='cuda') * 0.1
        B = torch.zeros(batch, seq_len, d_inner, 2, device='cuda')
        x = torch.zeros(batch, seq_len, d_inner, device='cuda')

        A_bar, _ = cayley_fused_pytorch(alpha, omega, dt, B, x)

        # For pure rotation: a11 = a22, a12 = -a21
        a11 = A_bar[..., 0]
        a12 = A_bar[..., 1]
        a21 = A_bar[..., 2]
        a22 = A_bar[..., 3]

        torch.testing.assert_close(a11, a22, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(a12, -a21, rtol=1e-5, atol=1e-5)


class TestStability:
    """Test A-stability guarantees."""

    def test_spectral_radius_bounded(self):
        """Verify |eigenvalues(A_bar)| <= 1 for all alpha >= 0."""
        torch.manual_seed(123)

        for _ in range(10):
            # Random positive alpha
            alpha = torch.abs(torch.randn(1000, device='cuda')) + 1e-6
            omega = torch.randn(1000, device='cuda') * 10
            dt = torch.rand(1000, device='cuda') * 0.1 + 0.001

            # Compute A_bar for each
            tau = dt / 2.0
            one_plus_tau_alpha = 1.0 + tau * alpha
            tau_omega = tau * omega
            det_M = one_plus_tau_alpha ** 2 + tau_omega ** 2
            inv_det = 1.0 / (det_M + 1e-6)

            one_minus_tau_alpha = 1.0 - tau * alpha

            # A_bar components
            m11 = one_plus_tau_alpha * inv_det
            m12 = tau_omega * inv_det
            n11 = one_minus_tau_alpha
            n12 = tau_omega

            a11 = m11 * n11 + m12 * (-tau_omega)
            a12 = m11 * n12 + m12 * n11

            # For 2x2 rotation-like matrix, eigenvalues are:
            # lambda = a11 +/- i * |a12|
            # |lambda|^2 = a11^2 + a12^2

            spectral_radius_sq = a11 ** 2 + a12 ** 2

            # Should be <= 1 (with small tolerance for fp precision)
            assert (spectral_radius_sq <= 1.0 + 1e-4).all(), \
                f"Spectral radius > 1: max = {spectral_radius_sq.max().item()}"

    def test_unitary_when_no_damping(self):
        """Test that A_bar is unitary when alpha = 0."""
        batch, seq_len, d_inner = 2, 16, 32

        alpha = torch.zeros(batch, seq_len, d_inner, device='cuda')
        omega = torch.randn(batch, seq_len, d_inner, device='cuda')
        dt = torch.rand(batch, seq_len, d_inner, device='cuda') * 0.1 + 0.001

        # Compute A_bar
        A_bar, _ = cayley_fused_pytorch(
            alpha, omega, dt,
            torch.zeros(batch, seq_len, d_inner, 2, device='cuda'),
            torch.zeros(batch, seq_len, d_inner, device='cuda'),
        )

        # Check |det(A_bar)| = 1 (unitary)
        a11, a12, a21, a22 = A_bar[..., 0], A_bar[..., 1], A_bar[..., 2], A_bar[..., 3]
        det = a11 * a22 - a12 * a21

        torch.testing.assert_close(
            det.abs(),
            torch.ones_like(det),
            rtol=1e-4, atol=1e-4,
        )

    def test_contractive_with_damping(self):
        """Test that A_bar is strictly contractive when alpha > 0."""
        batch, seq_len, d_inner = 2, 16, 32

        alpha = torch.ones(batch, seq_len, d_inner, device='cuda') * 0.5  # Positive damping
        omega = torch.randn(batch, seq_len, d_inner, device='cuda')
        dt = torch.ones(batch, seq_len, d_inner, device='cuda') * 0.1

        A_bar, _ = cayley_fused_pytorch(
            alpha, omega, dt,
            torch.zeros(batch, seq_len, d_inner, 2, device='cuda'),
            torch.zeros(batch, seq_len, d_inner, device='cuda'),
        )

        # Check |det(A_bar)| < 1 (contractive)
        a11, a12, a21, a22 = A_bar[..., 0], A_bar[..., 1], A_bar[..., 2], A_bar[..., 3]
        det = a11 * a22 - a12 * a21

        assert (det.abs() < 1.0).all(), "A_bar should be contractive with positive damping"


class TestCayleyAutograd:
    """Test autograd wrapper."""

    def test_backward_runs(self):
        """Test that backward pass runs without error."""
        torch.manual_seed(456)
        batch, seq_len, d_inner = 2, 32, 16

        alpha = torch.rand(batch, seq_len, d_inner, device='cuda', requires_grad=True)
        omega = torch.randn(batch, seq_len, d_inner, device='cuda', requires_grad=True)
        # Use leaf tensor for dt (multiply happens in separate step to keep as leaf)
        dt_raw = torch.rand(batch, seq_len, d_inner, device='cuda') * 0.1 + 0.001
        dt = dt_raw.clone().requires_grad_(True)
        B = torch.randn(batch, seq_len, d_inner, 2, device='cuda', requires_grad=True)
        x = torch.randn(batch, seq_len, d_inner, device='cuda', requires_grad=True)

        A_bar, u_bar = cayley_transform(alpha, omega, dt, B, x)
        loss = A_bar.sum() + u_bar.sum()
        loss.backward()

        assert alpha.grad is not None
        assert omega.grad is not None
        assert dt.grad is not None
        assert B.grad is not None
        assert x.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
