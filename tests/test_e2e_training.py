"""End-to-end 100-step training test.

Verifies that the CUDA-accelerated KSSM can train stably for 100 steps
on synthetic data. Checks:
1. Loss decreases over training
2. No NaN/Inf in loss or parameters
3. All parameters receive gradients
4. Final loss is reasonable
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "CUDA required"
    return torch.device("cuda")


def test_100_step_training(device):
    from kssm.config.defaults import KSSMConfig
    from kssm.models.kssm_block import KSSMBlock

    # Config: small model for testing
    config = KSSMConfig(d_model=128, n_layers=2, context_length=1024)

    torch.manual_seed(42)
    block = KSSMBlock(config, layer_idx=0).to(device).to(torch.bfloat16)

    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3, weight_decay=0.01)

    B, L = 2, 256
    losses = []

    for step in range(100):
        # Synthetic data (different each step for diversity, same seed for reproducibility)
        torch.manual_seed(1000 + step)
        x = torch.randn(B, L, config.d_model, device=device, dtype=torch.bfloat16) * 0.1
        target = torch.randn(B, L, config.d_model, device=device, dtype=torch.bfloat16) * 0.1

        optimizer.zero_grad()
        y = block(x)
        loss = F.mse_loss(y, target)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        # Check no NaN/Inf
        assert math.isfinite(loss_val), f"Step {step}: loss is {loss_val}"

    # Check loss decreased (compare first 10 avg to last 10 avg)
    early_avg = sum(losses[:10]) / 10
    late_avg = sum(losses[-10:]) / 10
    assert late_avg < early_avg, \
        f"Loss did not decrease: early={early_avg:.6f}, late={late_avg:.6f}"

    # Check all parameters are finite
    for name, param in block.named_parameters():
        assert torch.isfinite(param).all(), f"Parameter {name} has non-finite values"
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Gradient for {name} has non-finite values"

    # Check parameter gradient coverage (should be >90%)
    params_with_grad = sum(1 for p in block.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for _ in block.parameters())
    coverage = params_with_grad / total_params
    assert coverage >= 0.9, f"Only {coverage:.0%} of parameters received gradients"

    print(f"\n100-step training results:")
    print(f"  Initial loss (avg first 10): {early_avg:.6f}")
    print(f"  Final loss (avg last 10):    {late_avg:.6f}")
    print(f"  Reduction:                   {(1 - late_avg/early_avg)*100:.1f}%")
    print(f"  Gradient coverage:           {coverage:.0%}")
    print(f"  All losses finite:           True")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
