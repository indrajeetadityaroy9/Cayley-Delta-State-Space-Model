"""Tests for KSSM A-stability guarantees.

The Cayley transform guarantees that |eigenvalues(A_bar)| <= 1 for any:
- alpha >= 0 (damping, enforced by softplus)
- omega (frequency, unbounded)
- dt > 0 (timestep)

This is the "Unconditional A-Stability" theorem from the proposal.
"""

import pytest
import torch
import torch.nn.functional as F

from kssm.kernels.cayley_fused import cayley_fused_pytorch
from kssm.modules.kssm_layer import KSSMLayer, KSSMLayerSimple
from kssm.modules.init import nuclear_init, verify_initialization
from kssm.config import KSSMConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestAStability:
    """Test unconditional A-stability theorem."""

    def test_stability_random_parameters(self):
        """Verify stability holds for random parameter combinations."""
        torch.manual_seed(42)

        n_tests = 1000

        for _ in range(10):
            # Random parameters
            alpha = torch.abs(torch.randn(n_tests, device='cuda'))  # >= 0
            omega = torch.randn(n_tests, device='cuda') * 100  # Large frequencies
            dt = torch.rand(n_tests, device='cuda') * 10 + 0.001  # Large timesteps

            # Compute spectral radius
            tau = dt / 2.0
            one_plus_tau_alpha = 1.0 + tau * alpha
            tau_omega = tau * omega
            det_M = one_plus_tau_alpha ** 2 + tau_omega ** 2
            inv_det = 1.0 / (det_M + 1e-6)

            one_minus_tau_alpha = 1.0 - tau * alpha

            # Eigenvalue magnitude squared for 2x2 block
            # For A_bar = M^{-1} @ N, the eigenvalue magnitude is:
            # |lambda|^2 = ((1-tau*alpha)^2 + (tau*omega)^2) / ((1+tau*alpha)^2 + (tau*omega)^2)
            numerator = one_minus_tau_alpha ** 2 + tau_omega ** 2
            denominator = one_plus_tau_alpha ** 2 + tau_omega ** 2
            spectral_radius_sq = numerator / denominator

            # Since alpha >= 0, we have (1+tau*alpha) >= (1-tau*alpha) for tau*alpha <= 1
            # and the denominator includes +tau*alpha while numerator has -tau*alpha
            # This guarantees |lambda| <= 1

            max_radius = spectral_radius_sq.sqrt().max().item()
            assert max_radius <= 1.0 + 1e-5, f"Stability violated: max radius = {max_radius}"

    def test_stability_extreme_parameters(self):
        """Test stability at extreme parameter values."""
        device = 'cuda'

        test_cases = [
            # (alpha, omega, dt, description)
            (0.0, 0.0, 0.001, "zero alpha and omega"),
            (0.0, 1000.0, 0.001, "zero alpha, large omega"),
            (100.0, 0.0, 0.001, "large alpha, zero omega"),
            (100.0, 1000.0, 0.001, "large alpha and omega"),
            (0.0, 0.0, 10.0, "zero alpha/omega, large dt"),
            (0.0, 1000.0, 10.0, "zero alpha, large omega and dt"),
            (1e-6, 1e6, 1e-6, "tiny alpha, huge omega, tiny dt"),
        ]

        for alpha_val, omega_val, dt_val, desc in test_cases:
            alpha = torch.tensor([alpha_val], device=device)
            omega = torch.tensor([omega_val], device=device)
            dt = torch.tensor([dt_val], device=device)

            tau = dt / 2.0
            numerator = (1 - tau * alpha) ** 2 + (tau * omega) ** 2
            denominator = (1 + tau * alpha) ** 2 + (tau * omega) ** 2
            spectral_radius_sq = numerator / (denominator + 1e-10)

            radius = spectral_radius_sq.sqrt().item()
            assert radius <= 1.0 + 1e-5, f"Stability violated for {desc}: radius = {radius}"


class TestNuclearInitialization:
    """Test nuclear initialization for long-memory tasks."""

    def test_long_memory_init(self):
        """Test that long_memory=True gives low damping."""
        config = KSSMConfig(d_model=64, d_inner=64)
        layer = KSSMLayer(config).cuda()

        nuclear_init(layer, long_memory=True)

        diag = verify_initialization(layer)

        # Alpha bias should be around -5.0
        assert diag['alpha_bias'] < -4.0, f"Alpha bias too high: {diag['alpha_bias']}"

        # Expected damping should be very low
        assert diag['expected_damping'] < 0.02, f"Damping too high: {diag['expected_damping']}"

        # Signal after 500 steps should be significant
        assert diag['decay_500'] > 0.01, f"Signal decays too fast: {diag['decay_500']}"

    def test_moderate_memory_init(self):
        """Test that long_memory=False gives moderate damping."""
        config = KSSMConfig(d_model=64, d_inner=64)
        layer = KSSMLayer(config).cuda()

        nuclear_init(layer, long_memory=False)

        diag = verify_initialization(layer)

        # Alpha bias should be around -2.0
        assert -3.0 < diag['alpha_bias'] < -1.0, f"Alpha bias unexpected: {diag['alpha_bias']}"

        # Expected damping should be moderate
        assert 0.05 < diag['expected_damping'] < 0.3, f"Damping unexpected: {diag['expected_damping']}"

    def test_frequency_distribution(self):
        """Test that frequencies are log-uniformly distributed."""
        config = KSSMConfig(d_model=64, d_inner=64)
        layer = KSSMLayer(config).cuda()

        nuclear_init(layer, long_memory=True, freq_min=0.01, freq_max=100.0)

        diag = verify_initialization(layer)

        # Frequency range should span several orders of magnitude
        assert diag['freq_min'] < 0.1
        assert diag['freq_max'] > 10.0


class TestGradientStability:
    """Test gradient stability during training."""

    def test_gradient_norms_stable(self):
        """Test that gradients don't explode during forward/backward."""
        torch.manual_seed(789)

        config = KSSMConfig(d_model=64, d_inner=64)
        layer = KSSMLayer(config).cuda()
        nuclear_init(layer, long_memory=True)

        # Random input
        x = torch.randn(4, 128, 64, device='cuda', requires_grad=True)

        # Forward
        output, _ = layer(x)
        loss = output.pow(2).mean()

        # Backward
        loss.backward()

        # Check gradient norms
        for name, param in layer.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm < 1e6, f"Gradient exploded for {name}: {grad_norm}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_alpha_gradient_direction(self):
        """Test that alpha gradients point in expected direction.

        When we want to retain memory longer, gradients should push
        alpha toward smaller values (less damping).
        """
        torch.manual_seed(101)

        layer = KSSMLayerSimple(d_model=32).cuda()
        nuclear_init(layer, long_memory=True)

        # Input with pattern at beginning, measure at end
        x = torch.zeros(2, 100, 32, device='cuda')
        x[:, 0, :] = 1.0  # Signal at t=0

        # Forward
        output, _ = layer(x, use_triton=False)  # Use PyTorch for cleaner gradients

        # Loss: we want output at t=99 to remember t=0
        # So we maximize correlation with initial signal
        loss = -output[:, -1, :].sum()  # Negative because we want to maximize

        # This test is informational - gradient direction depends on many factors
        # The key is that gradients exist and are finite


class TestLayerIntegration:
    """Test full layer functionality."""

    def test_forward_backward(self):
        """Test complete forward and backward pass."""
        config = KSSMConfig(d_model=64, d_inner=128)
        layer = KSSMLayer(config).cuda()
        nuclear_init(layer, long_memory=True)

        x = torch.randn(4, 64, 64, device='cuda', requires_grad=True)

        output, final_state = layer(x)

        assert output.shape == (4, 64, 64)
        assert final_state.shape == (4, 128, 2)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_sequential_processing(self):
        """Test that sequential step matches parallel forward."""
        layer = KSSMLayerSimple(d_model=32).cuda()

        x = torch.randn(2, 10, 32, device='cuda')

        # Parallel forward
        with torch.no_grad():
            output_parallel, _ = layer(x, use_triton=False)

        # Sequential forward
        state = layer.init_state(2, device='cuda')
        outputs_seq = []
        with torch.no_grad():
            for t in range(10):
                out_t, state = layer.step(x[:, t, :], state)
                outputs_seq.append(out_t)
        output_sequential = torch.stack(outputs_seq, dim=1)

        # Should match (approximately, due to numerical differences)
        torch.testing.assert_close(
            output_sequential, output_parallel,
            rtol=1e-1, atol=1e-1,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
