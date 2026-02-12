"""Gradient correctness tests for CUDA autograd.Function wrappers.

Since the CUDA kernels use BF16 I/O (hardcoded reinterpret_cast to __nv_bfloat16),
we cannot use torch.autograd.gradcheck with FP64 inputs. Instead, we compare
gradients from the CUDA backward pass against finite-difference approximations
computed in BF16 precision.

For each kernel, we:
1. Run forward + backward through the autograd wrapper
2. Compute finite-difference gradients with a BF16-appropriate epsilon
3. Assert approximate agreement (BF16 tolerance)
"""

import math
import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "CUDA required"
    return torch.device("cuda")


def finite_diff_grad(fn, inputs, input_idx, eps=1e-2):
    """Compute finite-difference gradient for a single input tensor.

    Uses central differences: df/dx_i ≈ (f(x+eps*e_i) - f(x-eps*e_i)) / (2*eps)
    Only samples a subset of elements for tractability.
    """
    x = inputs[input_idx]
    grad = torch.zeros_like(x, dtype=torch.float32)
    flat = x.reshape(-1)
    n = min(flat.numel(), 50)  # Sample up to 50 elements

    torch.manual_seed(0)
    indices = torch.randperm(flat.numel(), device=x.device)[:n]

    for idx in indices:
        idx_item = idx.item()
        orig = flat[idx_item].item()

        # +eps
        flat[idx_item] = orig + eps
        inputs_plus = list(inputs)
        inputs_plus[input_idx] = flat.reshape(x.shape).to(x.dtype)
        out_plus = fn(*inputs_plus)
        if isinstance(out_plus, tuple):
            loss_plus = sum(o.float().sum() for o in out_plus)
        else:
            loss_plus = out_plus.float().sum()

        # -eps
        flat[idx_item] = orig - eps
        inputs_minus = list(inputs)
        inputs_minus[input_idx] = flat.reshape(x.shape).to(x.dtype)
        out_minus = fn(*inputs_minus)
        if isinstance(out_minus, tuple):
            loss_minus = sum(o.float().sum() for o in out_minus)
        else:
            loss_minus = out_minus.float().sum()

        # Restore
        flat[idx_item] = orig

        fd = (loss_plus.item() - loss_minus.item()) / (2 * eps)
        grad.reshape(-1)[idx_item] = fd

    return grad, indices


def compare_grads(autograd_grad, fd_grad, indices, name, atol=0.5, rtol=0.3):
    """Compare autograd and finite-difference gradients at sampled indices."""
    ag_flat = autograd_grad.float().reshape(-1)
    fd_flat = fd_grad.float().reshape(-1)

    ag_vals = ag_flat[indices]
    fd_vals = fd_flat[indices]

    # Use relative error where gradients are significant, absolute otherwise
    abs_diff = (ag_vals - fd_vals).abs()
    scale = torch.max(ag_vals.abs(), fd_vals.abs()).clamp(min=1e-6)
    rel_diff = abs_diff / scale

    # Pass if either absolute or relative error is within tolerance
    passed = (abs_diff < atol) | (rel_diff < rtol)
    pass_rate = passed.float().mean().item()

    assert pass_rate >= 0.8, \
        f"{name}: gradient agreement {pass_rate:.1%} (need >=80%). " \
        f"Max abs diff: {abs_diff.max():.4f}, max rel diff: {rel_diff.max():.4f}"


class TestCayleyVPGradients:
    """Gradient tests for CayleyVPFn."""

    def test_backward_with_gate(self, device):
        from cdssm.ops import cayley_vp_cuda

        B, L, H = 1, 32, 4
        torch.manual_seed(42)

        alpha = (torch.rand(B, L, H, device=device, dtype=torch.bfloat16) * 2.0).requires_grad_(True)
        omega = torch.randn(B, L, H, device=device, dtype=torch.bfloat16).requires_grad_(True)
        dt = (torch.rand(B, L, H, device=device, dtype=torch.bfloat16) * 0.5 + 0.1).requires_grad_(True)
        r_gate = torch.sigmoid(torch.randn(B, L, H, device=device, dtype=torch.bfloat16)).requires_grad_(True)
        gating_c = math.log(8192)

        # Autograd backward
        A_bar, vp = cayley_vp_cuda(alpha, omega, dt, r_gate, gating_c)
        loss = A_bar.float().sum() + vp.float().sum()
        loss.backward()

        ag_grads = {
            'alpha': alpha.grad.clone(),
            'omega': omega.grad.clone(),
            'dt': dt.grad.clone(),
            'r_gate': r_gate.grad.clone(),
        }

        # Finite differences for each input
        for name, inp_idx in [('alpha', 0), ('omega', 1), ('dt', 2), ('r_gate', 3)]:
            inputs = [alpha.detach(), omega.detach(), dt.detach(), r_gate.detach()]
            fd_grad, indices = finite_diff_grad(
                lambda a, w, d, r: cayley_vp_cuda(a, w, d, r, gating_c),
                inputs, inp_idx, eps=5e-3,
            )
            compare_grads(ag_grads[name], fd_grad, indices, f"cayley_vp/{name}")


class TestAdaptiveDtGradients:
    """Gradient tests for AdaptiveDtFn."""

    def test_backward(self, device):
        from cdssm.ops import adaptive_dt_cuda

        B, L, H = 1, 32, 4
        torch.manual_seed(42)

        alpha = F.softplus(torch.randn(B, L, H, device=device, dtype=torch.bfloat16)).requires_grad_(True)
        omega = torch.randn(B, L, H, device=device, dtype=torch.bfloat16).requires_grad_(True)
        log_dt_scale = torch.randn(H, device=device, dtype=torch.float32).requires_grad_(True)

        eps_bf16 = torch.finfo(torch.bfloat16).eps
        omega_thresh = math.sqrt(eps_bf16)
        delta = 16.0 * eps_bf16
        smoothness = omega_thresh / 5.0
        eps = eps_bf16 * 100

        # Autograd backward
        dt_out = adaptive_dt_cuda(alpha, omega, log_dt_scale, omega_thresh, delta, smoothness, eps)
        loss = dt_out.float().sum()
        loss.backward()

        ag_grads = {
            'alpha': alpha.grad.clone(),
            'omega': omega.grad.clone(),
            'log_dt_scale': log_dt_scale.grad.clone(),
        }

        # Finite differences
        for name, inp_idx in [('alpha', 0), ('omega', 1), ('log_dt_scale', 2)]:
            inputs = [alpha.detach(), omega.detach(), log_dt_scale.detach()]
            fd_eps = 1e-3 if name == 'log_dt_scale' else 5e-3
            fd_grad, indices = finite_diff_grad(
                lambda a, w, s: adaptive_dt_cuda(a, w, s, omega_thresh, delta, smoothness, eps),
                inputs, inp_idx, eps=fd_eps,
            )
            compare_grads(ag_grads[name], fd_grad, indices, f"adaptive_dt/{name}")


class TestIntraChunkScanGradients:
    """Gradient tests for IntraChunkScanFn."""

    def test_backward(self, device):
        from cdssm.ops import intra_chunk_scan_cuda

        BNC, C, H, D = 2, 8, 2, 16
        torch.manual_seed(42)

        # Well-conditioned A_bar (rotation + mild decay)
        theta = torch.randn(BNC, C, H, device=device) * 0.3
        decay = torch.rand(BNC, C, H, device=device) * 0.1 + 0.9
        cos_t = torch.cos(theta) * decay
        sin_t = torch.sin(theta) * decay

        A_flat = torch.zeros(BNC, C, H, 2, 2, device=device, dtype=torch.bfloat16)
        A_flat[..., 0, 0] = cos_t.bfloat16()
        A_flat[..., 0, 1] = sin_t.bfloat16()
        A_flat[..., 1, 0] = (-sin_t).bfloat16()
        A_flat[..., 1, 1] = cos_t.bfloat16()
        A_flat = A_flat.requires_grad_(True)

        K_flat = F.normalize(
            torch.randn(BNC, C, H, D, device=device).float(), dim=-1
        ).bfloat16().requires_grad_(True)
        V_flat = (torch.randn(BNC, C, H, 2, device=device, dtype=torch.bfloat16) * 0.1).requires_grad_(True)
        beta_flat = torch.sigmoid(
            torch.randn(BNC, C, H, device=device, dtype=torch.bfloat16)
        ).requires_grad_(True)

        # Autograd backward
        local_h, cum_A = intra_chunk_scan_cuda(A_flat, K_flat, V_flat, beta_flat)
        loss = local_h.float().sum() + cum_A.float().sum()
        loss.backward()

        ag_grads = {
            'A': A_flat.grad.clone(),
            'K': K_flat.grad.clone(),
            'V': V_flat.grad.clone(),
            'beta': beta_flat.grad.clone(),
        }

        # Finite differences
        for name, inp_idx in [('A', 0), ('K', 1), ('V', 2), ('beta', 3)]:
            inputs = [A_flat.detach(), K_flat.detach(), V_flat.detach(), beta_flat.detach()]
            fd_grad, indices = finite_diff_grad(
                lambda a, k, v, b: intra_chunk_scan_cuda(a, k, v, b),
                inputs, inp_idx, eps=5e-3,
            )
            compare_grads(ag_grads[name], fd_grad, indices, f"intra_chunk/{name}",
                         atol=1.0, rtol=0.5)  # Relaxed for sequential accumulation


class TestInterChunkScanGradients:
    """Gradient tests for InterChunkScanFn."""

    def test_backward(self, device):
        from cdssm.ops import inter_chunk_scan_cuda

        B, NC, H, D = 1, 4, 2, 16
        torch.manual_seed(42)

        theta = torch.randn(B, NC, H, device=device) * 0.3
        decay = torch.rand(B, NC, H, device=device) * 0.1 + 0.9
        cos_t = torch.cos(theta) * decay
        sin_t = torch.sin(theta) * decay

        total_A = torch.zeros(B, NC, H, 2, 2, device=device, dtype=torch.bfloat16)
        total_A[..., 0, 0] = cos_t.bfloat16()
        total_A[..., 0, 1] = sin_t.bfloat16()
        total_A[..., 1, 0] = (-sin_t).bfloat16()
        total_A[..., 1, 1] = cos_t.bfloat16()
        total_A = total_A.requires_grad_(True)

        final_local_h = (
            torch.randn(B, NC, H, 2, D, device=device, dtype=torch.bfloat16) * 0.1
        ).requires_grad_(True)

        # Autograd backward
        cs = inter_chunk_scan_cuda(total_A, final_local_h)
        loss = cs.float().sum()
        loss.backward()

        ag_grads = {
            'total_A': total_A.grad.clone(),
            'final_local_h': final_local_h.grad.clone(),
        }

        # Finite differences
        for name, inp_idx in [('total_A', 0), ('final_local_h', 1)]:
            inputs = [total_A.detach(), final_local_h.detach()]
            fd_grad, indices = finite_diff_grad(
                inter_chunk_scan_cuda,
                inputs, inp_idx, eps=5e-3,
            )
            compare_grads(ag_grads[name], fd_grad, indices, f"inter_chunk/{name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
