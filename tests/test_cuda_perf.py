"""Performance benchmarks for CUDA kernels using torch.cuda.Event timers.

Benchmarks each kernel individually (forward + backward) and the full
CDSSMBlock forward+backward pass.

Shapes: B=4, L=1024, H=12, D=64 (default CDSSM config with d_model=384).
"""

import math
import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "CUDA required"
    return torch.device("cuda")


def benchmark_fn(fn, warmup=10, iters=50):
    """Benchmark a function with CUDA event timers.

    Returns: median time in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]  # median


class TestKernelPerformance:
    """Per-kernel forward+backward timing."""

    def test_cayley_vp_perf(self, device):
        from cdssm.ops import cayley_vp_cuda

        B, L, H = 4, 1024, 12
        gating_c = math.log(8192)

        alpha = (torch.rand(B, L, H, device=device, dtype=torch.bfloat16) * 2.0).requires_grad_(True)
        omega = torch.randn(B, L, H, device=device, dtype=torch.bfloat16).requires_grad_(True)
        dt = (torch.rand(B, L, H, device=device, dtype=torch.bfloat16) * 0.5 + 0.01).requires_grad_(True)
        r_gate = torch.sigmoid(torch.randn(B, L, H, device=device, dtype=torch.bfloat16)).requires_grad_(True)

        def fwd_bwd():
            A, vp = cayley_vp_cuda(alpha, omega, dt, r_gate, gating_c)
            (A.sum() + vp.sum()).backward()

        t = benchmark_fn(fwd_bwd)
        print(f"\ncayley_vp fwd+bwd: {t:.3f} ms (B={B}, L={L}, H={H})")

    def test_adaptive_dt_perf(self, device):
        from cdssm.ops import adaptive_dt_cuda

        B, L, H = 4, 1024, 12
        eps_bf16 = torch.finfo(torch.bfloat16).eps
        omega_thresh = math.sqrt(eps_bf16)
        delta = 16.0 * eps_bf16
        smoothness = omega_thresh / 5.0
        eps = eps_bf16 * 100

        alpha = F.softplus(torch.randn(B, L, H, device=device, dtype=torch.bfloat16)).requires_grad_(True)
        omega = torch.randn(B, L, H, device=device, dtype=torch.bfloat16).requires_grad_(True)
        log_dt_scale = torch.randn(H, device=device, dtype=torch.float32).requires_grad_(True)

        def fwd_bwd():
            dt = adaptive_dt_cuda(alpha, omega, log_dt_scale, omega_thresh, delta, smoothness, eps)
            dt.sum().backward()

        t = benchmark_fn(fwd_bwd)
        print(f"\nadaptive_dt fwd+bwd: {t:.3f} ms (B={B}, L={L}, H={H})")

    def test_intra_chunk_scan_perf(self, device):
        from cdssm.ops import intra_chunk_scan_cuda

        B, L, H, D = 4, 1024, 12, 64
        C = 64
        BNC = B * (L // C)

        theta = torch.randn(BNC, C, H, device=device) * 0.3
        decay = torch.rand(BNC, C, H, device=device) * 0.1 + 0.9
        A_flat = torch.zeros(BNC, C, H, 2, 2, device=device, dtype=torch.bfloat16)
        A_flat[..., 0, 0] = (torch.cos(theta) * decay).bfloat16()
        A_flat[..., 0, 1] = (torch.sin(theta) * decay).bfloat16()
        A_flat[..., 1, 0] = (-torch.sin(theta) * decay).bfloat16()
        A_flat[..., 1, 1] = (torch.cos(theta) * decay).bfloat16()
        A_flat = A_flat.requires_grad_(True)

        K_flat = F.normalize(
            torch.randn(BNC, C, H, D, device=device).float(), dim=-1
        ).bfloat16().requires_grad_(True)
        V_flat = (torch.randn(BNC, C, H, 2, device=device, dtype=torch.bfloat16) * 0.1).requires_grad_(True)
        beta_flat = torch.sigmoid(
            torch.randn(BNC, C, H, device=device, dtype=torch.bfloat16)
        ).requires_grad_(True)

        def fwd_bwd():
            lh, cA = intra_chunk_scan_cuda(A_flat, K_flat, V_flat, beta_flat)
            (lh.sum() + cA.sum()).backward()

        t = benchmark_fn(fwd_bwd)
        print(f"\nintra_chunk_scan fwd+bwd: {t:.3f} ms (BNC={BNC}, C={C}, H={H}, D={D})")

    def test_inter_chunk_scan_perf(self, device):
        from cdssm.ops import inter_chunk_scan_cuda

        B, NC, H, D = 4, 16, 12, 64

        theta = torch.randn(B, NC, H, device=device) * 0.3
        decay = torch.rand(B, NC, H, device=device) * 0.1 + 0.9
        total_A = torch.zeros(B, NC, H, 2, 2, device=device, dtype=torch.bfloat16)
        total_A[..., 0, 0] = (torch.cos(theta) * decay).bfloat16()
        total_A[..., 0, 1] = (torch.sin(theta) * decay).bfloat16()
        total_A[..., 1, 0] = (-torch.sin(theta) * decay).bfloat16()
        total_A[..., 1, 1] = (torch.cos(theta) * decay).bfloat16()
        total_A = total_A.requires_grad_(True)

        final_local_h = (
            torch.randn(B, NC, H, 2, D, device=device, dtype=torch.bfloat16) * 0.1
        ).requires_grad_(True)

        def fwd_bwd():
            cs = inter_chunk_scan_cuda(total_A, final_local_h)
            cs.sum().backward()

        t = benchmark_fn(fwd_bwd)
        print(f"\ninter_chunk_scan fwd+bwd: {t:.3f} ms (B={B}, NC={NC}, H={H}, D={D})")


class TestBlockPerformance:
    """Full CDSSMBlock forward+backward timing."""

    def test_full_block_perf(self, device):
        from cdssm.config import CDSSMConfig
        from cdssm.models.block import CDSSMBlock

        config = CDSSMConfig(d_model=384, n_layers=12, context_length=8192)
        B, L = 4, 1024

        torch.manual_seed(42)
        block = CDSSMBlock(config, layer_idx=0).to(device).to(torch.bfloat16)
        x = torch.randn(B, L, config.d_model, device=device, dtype=torch.bfloat16)

        # Forward only
        def fwd_only():
            y = block(x)
            return y

        t_fwd = benchmark_fn(fwd_only)

        # Forward + backward
        x_grad = x.clone().requires_grad_(True)

        def fwd_bwd():
            y = block(x_grad)
            y.sum().backward()

        t_fwd_bwd = benchmark_fn(fwd_bwd)

        print(f"\nCDSSMBlock (d_model={config.d_model}, H={config.n_heads}, D={config.head_dim}):")
        print(f"  Forward:          {t_fwd:.3f} ms  (B={B}, L={L})")
        print(f"  Forward+Backward: {t_fwd_bwd:.3f} ms  (B={B}, L={L})")
        print(f"  Throughput:       {B * L / (t_fwd_bwd / 1000):.0f} tokens/sec")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
