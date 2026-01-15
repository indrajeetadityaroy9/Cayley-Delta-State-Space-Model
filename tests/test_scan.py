"""Tests for KSSM evolution (scan) operations.

Verifies that Triton kernel matches PyTorch float64 reference within tolerance.
"""

import pytest
import torch

from kssm.kernels.evolution_fwd import evolution_fwd, evolution_fwd_with_initial
from kssm.ops.reference import evolution_reference
from kssm.ops.scan_op import evolution


# Skip tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestEvolutionForward:
    """Test forward evolution kernel."""

    @pytest.mark.parametrize("batch,seq_len,d_inner", [
        (1, 16, 32),      # Small
        (4, 64, 64),      # Medium
        (8, 256, 128),    # Larger
        (2, 512, 256),    # Long sequence
    ])
    def test_triton_matches_reference(self, batch, seq_len, d_inner):
        """Verify Triton kernel matches PyTorch reference."""
        torch.manual_seed(42)

        # Generate random A_bar with spectral radius <= 1 (A-stable)
        # Using rotation-like matrices for realism
        theta = torch.rand(batch, seq_len, d_inner, device='cuda') * 2 * 3.14159
        scale = torch.rand(batch, seq_len, d_inner, device='cuda') * 0.9 + 0.05  # 0.05 to 0.95

        # Construct A_bar as scaled rotation: [[c*cos, -c*sin], [c*sin, c*cos]]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        A_bar = torch.stack([
            scale * cos_t,   # a11
            -scale * sin_t,  # a12
            scale * sin_t,   # a21
            scale * cos_t,   # a22
        ], dim=-1)  # (batch, seq, d_inner, 4)

        # Random u_bar
        u_bar = torch.randn(batch, seq_len, d_inner, 2, device='cuda')

        # Reference (float64)
        states_ref = evolution_reference(A_bar, u_bar)

        # Triton (bf16)
        A_bar_bf16 = A_bar.bfloat16().contiguous()
        u_bar_bf16 = u_bar.bfloat16().contiguous()
        states_triton = evolution_fwd(A_bar_bf16, u_bar_bf16)

        # Convert to same dtype for comparison
        states_ref = states_ref.float()
        states_triton = states_triton.float()

        # Check shapes match
        assert states_triton.shape == states_ref.shape

        # Check values match within tolerance
        # bf16 has ~3 decimal digits of precision, and error accumulates over sequence
        # rtol=5e-2, atol=5e-2 accounts for accumulated bf16 precision loss
        torch.testing.assert_close(
            states_triton, states_ref,
            rtol=5e-2, atol=5e-2,
            msg=f"Triton kernel mismatch for shape ({batch}, {seq_len}, {d_inner})"
        )

    def test_with_initial_state(self):
        """Test evolution with non-zero initial state."""
        torch.manual_seed(123)
        batch, seq_len, d_inner = 4, 64, 64  # Shorter sequence for tighter bounds

        # Stable A_bar
        scale = 0.9
        A_bar = torch.zeros(batch, seq_len, d_inner, 4, device='cuda')
        A_bar[..., 0] = scale  # a11
        A_bar[..., 3] = scale  # a22 (diagonal, no rotation)

        u_bar = torch.randn(batch, seq_len, d_inner, 2, device='cuda') * 0.1  # Smaller inputs
        initial_state = torch.randn(batch, d_inner, 2, device='cuda')

        # Reference
        states_ref = evolution_reference(A_bar, u_bar, initial_state)

        # Triton
        states_triton = evolution_fwd_with_initial(
            A_bar.bfloat16(),
            u_bar.bfloat16(),
            initial_state.bfloat16(),
        )

        torch.testing.assert_close(
            states_triton.float(), states_ref.float(),
            rtol=1e-1, atol=1e-1,  # Wider tolerance for accumulated bf16 error
        )

    def test_zero_input(self):
        """Test with zero input - state should decay."""
        batch, seq_len, d_inner = 2, 100, 32

        # Contractive A_bar (scale < 1)
        scale = 0.8
        A_bar = torch.zeros(batch, seq_len, d_inner, 4, device='cuda')
        A_bar[..., 0] = scale
        A_bar[..., 3] = scale

        # Zero input
        u_bar = torch.zeros(batch, seq_len, d_inner, 2, device='cuda')
        initial_state = torch.ones(batch, d_inner, 2, device='cuda')

        states = evolution_fwd_with_initial(
            A_bar.bfloat16(),
            u_bar.bfloat16(),
            initial_state.bfloat16(),
        )

        # Final state should be close to scale^seq_len * initial
        expected_final = initial_state * (scale ** seq_len)

        torch.testing.assert_close(
            states[:, -1, :, :].float(),
            expected_final.float(),
            rtol=5e-2, atol=5e-2,
        )

    def test_identity_matrix(self):
        """Test with identity A_bar - state should accumulate inputs."""
        batch, seq_len, d_inner = 2, 50, 32

        # Identity A_bar
        A_bar = torch.zeros(batch, seq_len, d_inner, 4, device='cuda')
        A_bar[..., 0] = 1.0  # a11 = 1
        A_bar[..., 3] = 1.0  # a22 = 1

        # Constant input
        u_bar = torch.ones(batch, seq_len, d_inner, 2, device='cuda') * 0.1

        states = evolution_fwd(
            A_bar.bfloat16(),
            u_bar.bfloat16(),
        )

        # State at time t should be 0.1 * t
        for t in [10, 20, 49]:
            expected = 0.1 * (t + 1)  # +1 because we count from 0
            actual = states[:, t, :, :].float().mean().item()
            assert abs(actual - expected) < 0.02, f"t={t}: expected {expected}, got {actual}"


class TestEvolutionAutograd:
    """Test autograd wrapper."""

    def test_autograd_forward(self):
        """Test that autograd wrapper produces same results."""
        torch.manual_seed(456)
        batch, seq_len, d_inner = 4, 64, 32

        A_bar = torch.randn(batch, seq_len, d_inner, 4, device='cuda')
        # Make A_bar stable
        A_bar = A_bar * 0.3

        u_bar = torch.randn(batch, seq_len, d_inner, 2, device='cuda')

        # Triton via wrapper
        states_triton = evolution(A_bar, u_bar, use_triton=True)

        # Reference
        states_ref = evolution_reference(A_bar, u_bar)

        torch.testing.assert_close(
            states_triton.float(), states_ref.float(),
            rtol=5e-2, atol=5e-2,
        )

    def test_backward_runs(self):
        """Test that backward pass runs without error."""
        torch.manual_seed(789)
        batch, seq_len, d_inner = 2, 32, 16

        A_bar = torch.randn(batch, seq_len, d_inner, 4, device='cuda', requires_grad=True)
        u_bar = torch.randn(batch, seq_len, d_inner, 2, device='cuda', requires_grad=True)

        states = evolution(A_bar * 0.3, u_bar)
        loss = states.sum()
        loss.backward()

        assert A_bar.grad is not None
        assert u_bar.grad is not None
        assert not torch.isnan(A_bar.grad).any()
        assert not torch.isnan(u_bar.grad).any()

    def test_gradient_finite(self):
        """Test that gradients are finite for reasonable inputs."""
        torch.manual_seed(101)
        batch, seq_len, d_inner = 4, 128, 64

        # Create A_bar directly with requires_grad
        A_bar = torch.randn(batch, seq_len, d_inner, 4, device='cuda', requires_grad=True)
        u_bar = torch.randn(batch, seq_len, d_inner, 2, device='cuda', requires_grad=True)

        # Scale down to ensure stability
        states = evolution(A_bar * 0.3, u_bar * 0.1)
        loss = states.pow(2).mean()
        loss.backward()

        assert A_bar.grad is not None, "A_bar gradient is None"
        assert u_bar.grad is not None, "u_bar gradient is None"
        assert torch.isfinite(A_bar.grad).all(), "A_bar gradient has non-finite values"
        assert torch.isfinite(u_bar.grad).all(), "u_bar gradient has non-finite values"


class TestNumericalPrecision:
    """Test numerical precision over long sequences."""

    def test_long_sequence_stability(self):
        """Test that long sequences don't cause catastrophic precision issues."""
        batch, seq_len, d_inner = 2, 4096, 64

        # Use unitary A_bar (rotation only, no decay)
        theta = torch.ones(batch, seq_len, d_inner, device='cuda') * 0.01  # Small rotation
        A_bar = torch.stack([
            torch.cos(theta),
            -torch.sin(theta),
            torch.sin(theta),
            torch.cos(theta),
        ], dim=-1)

        u_bar = torch.zeros(batch, seq_len, d_inner, 2, device='cuda')
        initial_state = torch.ones(batch, d_inner, 2, device='cuda')

        states = evolution_fwd_with_initial(
            A_bar.bfloat16(),
            u_bar.bfloat16(),
            initial_state.bfloat16(),
        )

        # For unitary rotation, magnitude should be preserved
        # bf16 accumulates error, so we allow more drift over 4096 steps
        final_magnitude = states[:, -1, :, :].float().norm(dim=-1).mean()
        initial_magnitude = initial_state.norm(dim=-1).mean()

        # bf16 with 4096 steps can have significant drift, but shouldn't explode/vanish
        ratio = final_magnitude / initial_magnitude
        assert 0.5 < ratio < 2.0, f"Magnitude drifted catastrophically: ratio={ratio}"

    def test_contractive_sequence(self):
        """Test that contractive dynamics properly decay."""
        batch, seq_len, d_inner = 2, 1000, 32

        # Strongly contractive
        scale = 0.99
        A_bar = torch.zeros(batch, seq_len, d_inner, 4, device='cuda')
        A_bar[..., 0] = scale
        A_bar[..., 3] = scale

        u_bar = torch.zeros(batch, seq_len, d_inner, 2, device='cuda')
        initial_state = torch.ones(batch, d_inner, 2, device='cuda')

        states = evolution_fwd_with_initial(
            A_bar.bfloat16(),
            u_bar.bfloat16(),
            initial_state.bfloat16(),
        )

        # After 1000 steps with scale=0.99: 0.99^1000 ≈ 4.3e-5
        expected_decay = scale ** seq_len
        actual = states[:, -1, :, :].float().abs().mean().item()
        initial_val = initial_state.abs().mean().item()

        assert actual / initial_val < 0.01, f"Decay insufficient: {actual / initial_val}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
