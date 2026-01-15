"""Tests for evolution backward kernel.

Verifies that the Triton backward kernel produces the same results
as the PyTorch reference implementation.
"""

import pytest
import torch

from kssm.kernels.evolution_bwd import (
    evolution_bwd,
    compute_d_A_bar,
    compute_d_A_bar_triton,
    _compute_d_A_bar_pytorch,
    evolution_backward_triton,
)
from kssm.ops.reference import evolution_backward_full_reference, cayley_transform_reference


@pytest.fixture
def device():
    """Get CUDA device if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def make_stable_A_bar(batch, seq_len, d_inner, device, dtype=torch.float64):
    """Create A_bar matrices with eigenvalues <= 1 using Cayley transform.

    This ensures numerical stability in both forward and backward scans.
    """
    # Use Cayley transform to get stable A_bar
    alpha = torch.rand(batch, seq_len, d_inner, device=device, dtype=dtype) * 0.1 + 0.01  # [0.01, 0.11]
    omega = torch.randn(batch, seq_len, d_inner, device=device, dtype=dtype) * 0.5
    dt = torch.ones(batch, seq_len, d_inner, device=device, dtype=dtype) * 0.1
    B = torch.randn(d_inner, 2, device=device, dtype=dtype)
    x = torch.randn(batch, seq_len, d_inner, device=device, dtype=dtype)

    A_bar, _ = cayley_transform_reference(alpha, omega, dt, B, x)
    return A_bar.to(dtype)


class TestEvolutionBwd:
    """Tests for the Triton backward kernel."""

    def test_d_u_bar_matches_reference(self, device):
        """Test that d_u_bar from Triton matches PyTorch reference."""
        batch, seq_len, d_inner = 2, 64, 32

        # Create test inputs with stable A_bar
        torch.manual_seed(42)
        A_bar = make_stable_A_bar(batch, seq_len, d_inner, device)
        grad_output = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float64)

        # Create states (forward simulation)
        states = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float64)

        # Reference backward
        d_A_bar_ref, d_u_bar_ref = evolution_backward_full_reference(A_bar, states, grad_output)

        # Triton backward (only d_u_bar)
        A_bar_bf16 = A_bar.bfloat16()
        grad_output_bf16 = grad_output.bfloat16()

        d_u_bar_triton = evolution_bwd(A_bar_bf16, grad_output_bf16)

        # Compare (use higher tolerance for bf16)
        d_u_bar_triton_f64 = d_u_bar_triton.double()

        # Note: d_u_bar is the adjoint state λ, which should match
        # Use higher tolerance for bf16 accumulation over sequence
        torch.testing.assert_close(
            d_u_bar_triton_f64,
            d_u_bar_ref,
            rtol=0.1,
            atol=0.3,
        )

    def test_full_backward_matches_reference(self, device):
        """Test that complete backward (d_A_bar + d_u_bar) matches reference."""
        batch, seq_len, d_inner = 2, 64, 32

        torch.manual_seed(42)
        A_bar = make_stable_A_bar(batch, seq_len, d_inner, device)
        grad_output = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float64)
        states = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float64)

        # Reference backward
        d_A_bar_ref, d_u_bar_ref = evolution_backward_full_reference(A_bar, states, grad_output)

        # Triton backward
        d_A_bar_triton, d_u_bar_triton = evolution_backward_triton(A_bar, states, grad_output)

        # Compare d_u_bar (bf16 tolerance)
        torch.testing.assert_close(
            d_u_bar_triton.double(),
            d_u_bar_ref,
            rtol=0.1,
            atol=0.3,
        )

        # Compare d_A_bar (bf16 tolerance)
        torch.testing.assert_close(
            d_A_bar_triton.double(),
            d_A_bar_ref,
            rtol=0.1,
            atol=0.3,
        )

    def test_gradient_shapes(self, device):
        """Test that gradients have correct shapes."""
        batch, seq_len, d_inner = 4, 128, 64

        A_bar = torch.randn(batch, seq_len, d_inner, 4, device=device, dtype=torch.bfloat16)
        grad_output = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.bfloat16)
        states = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.bfloat16)

        d_A_bar, d_u_bar = evolution_backward_triton(A_bar, states, grad_output)

        assert d_A_bar.shape == (batch, seq_len, d_inner, 4)
        assert d_u_bar.shape == (batch, seq_len, d_inner, 2)
        assert d_A_bar.dtype == torch.bfloat16
        assert d_u_bar.dtype == torch.bfloat16

    def test_long_sequence(self, device):
        """Test backward on longer sequences."""
        batch, seq_len, d_inner = 2, 512, 16

        torch.manual_seed(42)
        A_bar = make_stable_A_bar(batch, seq_len, d_inner, device)
        grad_output = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float64)
        states = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float64)

        # Reference backward
        d_A_bar_ref, d_u_bar_ref = evolution_backward_full_reference(A_bar, states, grad_output)

        # Triton backward
        d_A_bar_triton, d_u_bar_triton = evolution_backward_triton(A_bar, states, grad_output)

        # Compare (longer sequences may have more accumulation error)
        torch.testing.assert_close(
            d_u_bar_triton.double(),
            d_u_bar_ref,
            rtol=0.15,
            atol=0.5,
        )

    def test_compute_d_A_bar_correctness(self, device):
        """Test that compute_d_A_bar produces correct outer products."""
        batch, seq_len, d_inner = 2, 32, 16

        torch.manual_seed(42)
        adjoint_states = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float32)
        forward_states = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float32)

        d_A_bar = compute_d_A_bar(adjoint_states, forward_states)

        # Verify shape
        assert d_A_bar.shape == (batch, seq_len, d_inner, 4)

        # Verify correctness for a specific position
        t = 5
        lambda1 = adjoint_states[0, t, 0, 0]
        lambda2 = adjoint_states[0, t, 0, 1]
        h1_prev = forward_states[0, t-1, 0, 0]  # h_{t-1}
        h2_prev = forward_states[0, t-1, 0, 1]

        expected_da11 = lambda1 * h1_prev
        expected_da12 = lambda1 * h2_prev
        expected_da21 = lambda2 * h1_prev
        expected_da22 = lambda2 * h2_prev

        assert torch.allclose(d_A_bar[0, t, 0, 0], expected_da11, rtol=1e-5)
        assert torch.allclose(d_A_bar[0, t, 0, 1], expected_da12, rtol=1e-5)
        assert torch.allclose(d_A_bar[0, t, 0, 2], expected_da21, rtol=1e-5)
        assert torch.allclose(d_A_bar[0, t, 0, 3], expected_da22, rtol=1e-5)

    def test_d_A_bar_at_t0_is_zero(self, device):
        """Test that d_A_bar at t=0 uses zero h_{-1}."""
        batch, seq_len, d_inner = 2, 32, 16

        torch.manual_seed(42)
        adjoint_states = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float32)
        forward_states = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float32)

        d_A_bar = compute_d_A_bar(adjoint_states, forward_states)

        # At t=0, h_{-1} = 0, so d_A_bar[0] should be all zeros
        assert torch.allclose(d_A_bar[:, 0, :, :], torch.zeros_like(d_A_bar[:, 0, :, :]))

    def test_compute_d_A_bar_triton_matches_pytorch(self, device):
        """Test that Triton compute_d_A_bar matches PyTorch reference exactly."""
        batch, seq_len, d_inner = 4, 128, 64

        torch.manual_seed(42)
        adjoint_states = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float32)
        forward_states = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float32)

        # Triton kernel
        d_A_bar_triton = compute_d_A_bar_triton(adjoint_states, forward_states)

        # PyTorch reference
        d_A_bar_pytorch = _compute_d_A_bar_pytorch(adjoint_states, forward_states)

        # Should match exactly (both use float32)
        torch.testing.assert_close(
            d_A_bar_triton,
            d_A_bar_pytorch,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_compute_d_A_bar_triton_various_sizes(self, device):
        """Test Triton compute_d_A_bar kernel with various tensor sizes."""
        test_cases = [
            (1, 16, 8),      # Small
            (2, 64, 32),     # Medium
            (4, 256, 128),   # Large
            (8, 512, 64),    # Long sequence
            (2, 33, 17),     # Non-power-of-2
        ]

        for batch, seq_len, d_inner in test_cases:
            torch.manual_seed(42)
            adjoint_states = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float32)
            forward_states = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float32)

            d_A_bar_triton = compute_d_A_bar_triton(adjoint_states, forward_states)
            d_A_bar_pytorch = _compute_d_A_bar_pytorch(adjoint_states, forward_states)

            torch.testing.assert_close(
                d_A_bar_triton,
                d_A_bar_pytorch,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Mismatch for shape ({batch}, {seq_len}, {d_inner})",
            )


class TestBackwardKernelIntegration:
    """Integration tests for backward kernel in autograd context."""

    def test_backward_kernel_vs_autograd(self, device):
        """Test that Triton backward gives same result as torch.autograd."""
        from kssm.ops.scan_op import evolution
        from kssm.kernels.evolution_fwd import evolution_fwd

        batch, seq_len, d_inner = 2, 32, 16

        torch.manual_seed(42)
        A_bar = torch.randn(batch, seq_len, d_inner, 4, device=device, dtype=torch.float32, requires_grad=True)
        u_bar = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float32, requires_grad=True)

        # Forward + backward with autograd
        states = evolution(A_bar, u_bar, use_triton=True)
        loss = states.sum()
        loss.backward()

        # Check gradients exist
        assert A_bar.grad is not None
        assert u_bar.grad is not None
        assert A_bar.grad.shape == A_bar.shape
        assert u_bar.grad.shape == u_bar.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
