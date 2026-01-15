"""Benchmark backward pass: Triton vs PyTorch reference."""

import torch
import time

from kssm.kernels.evolution_bwd import evolution_backward_triton
from kssm.ops.reference import evolution_backward_full_reference, cayley_transform_reference


def make_stable_A_bar(batch, seq_len, d_inner, device, dtype=torch.float64):
    """Create stable A_bar matrices using Cayley transform."""
    alpha = torch.rand(batch, seq_len, d_inner, device=device, dtype=dtype) * 0.1 + 0.01
    omega = torch.randn(batch, seq_len, d_inner, device=device, dtype=dtype) * 0.5
    dt = torch.ones(batch, seq_len, d_inner, device=device, dtype=dtype) * 0.1
    B = torch.randn(d_inner, 2, device=device, dtype=dtype)
    x = torch.randn(batch, seq_len, d_inner, device=device, dtype=dtype)

    A_bar, _ = cayley_transform_reference(alpha, omega, dt, B, x)
    return A_bar.to(dtype)


def benchmark_backward(batch, seq_len, d_inner, n_warmup=5, n_iter=20):
    """Benchmark Triton vs PyTorch backward pass."""
    device = "cuda"

    # Create test data
    torch.manual_seed(42)
    A_bar = make_stable_A_bar(batch, seq_len, d_inner, device)
    grad_output = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float64)
    states = torch.randn(batch, seq_len, d_inner, 2, device=device, dtype=torch.float64)

    # Warmup PyTorch
    for _ in range(n_warmup):
        _ = evolution_backward_full_reference(A_bar, states, grad_output)
    torch.cuda.synchronize()

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = evolution_backward_full_reference(A_bar, states, grad_output)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / n_iter * 1000

    # Warmup Triton
    for _ in range(n_warmup):
        _ = evolution_backward_triton(A_bar, states, grad_output)
    torch.cuda.synchronize()

    # Benchmark Triton
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = evolution_backward_triton(A_bar, states, grad_output)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iter * 1000

    speedup = pytorch_time / triton_time

    return pytorch_time, triton_time, speedup


def main():
    print("Backward Pass Benchmark: Triton vs PyTorch Reference")
    print("=" * 70)
    print(f"{'Config':<25} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    configs = [
        # (batch, seq_len, d_inner)
        (2, 256, 16),
        (4, 256, 32),
        (4, 512, 32),
        (8, 256, 64),
        (4, 1024, 32),
        (4, 2048, 32),
        (4, 4096, 32),
    ]

    for batch, seq_len, d_inner in configs:
        try:
            pytorch_ms, triton_ms, speedup = benchmark_backward(batch, seq_len, d_inner)
            config = f"B={batch}, L={seq_len}, D={d_inner}"
            print(f"{config:<25} {pytorch_ms:<15.3f} {triton_ms:<15.3f} {speedup:<10.2f}x")
        except Exception as e:
            config = f"B={batch}, L={seq_len}, D={d_inner}"
            print(f"{config:<25} ERROR: {e}")

    print("=" * 70)


if __name__ == "__main__":
    main()
