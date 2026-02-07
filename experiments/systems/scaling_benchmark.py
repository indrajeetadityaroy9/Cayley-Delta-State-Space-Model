"""Scaling Benchmark: Throughput & Memory vs Sequence Length.

Measures tokens/second and peak memory across sequence lengths.
Compares KSSM (O(L)) against a CausalTransformer baseline (O(L^2)).

Mamba-2 reference numbers from Dao & Gu (2024) cited for context.
"""

import gc
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from experiments.seed import seed_everything
from kssm.config import KSSMConfig
from kssm.model.backbone import KSSMBackbone


# ──────────────────────────────────────────────────────────────
# Transformer Baseline
# ──────────────────────────────────────────────────────────────

class CausalTransformer(nn.Module):
    """Causal Transformer using PyTorch 2.x SDPA (FlashAttention when available).

    Provides O(L^2) baseline for throughput comparison against KSSM's O(L) scaling.
    Uses pre-norm architecture with no dropout (benchmark mode).
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True, norm_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self._causal_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        if self._causal_mask is None or self._causal_mask.size(0) != L:
            self._causal_mask = torch.triu(
                torch.full((L, L), float('-inf'), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
        return self.encoder(x, mask=self._causal_mask, is_causal=True)


@dataclass
class BenchmarkResult:
    times_ms: list[float] = field(default_factory=list)
    memory_gb: float = 0.0

    @property
    def mean(self) -> float:
        return sum(self.times_ms) / len(self.times_ms) if self.times_ms else 0.0


def run_cuda_benchmark(fn, warmup: int = 5, iterations: int = 20) -> BenchmarkResult:
    """Benchmark with CUDA event timing."""
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(iterations)]
    memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return BenchmarkResult(times_ms=times_ms, memory_gb=memory_gb)


def benchmark_model(model, batch_size, seq_len, d_model, device: torch.device, warmup=5, iterations=20):
    """Benchmark forward+backward pass."""
    model = model.to(device).bfloat16()
    model.train()

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.bfloat16)

    def forward_backward():
        x_in = x.clone().requires_grad_(True)
        out = model(x_in)
        out.sum().backward()

    result = run_cuda_benchmark(forward_backward, warmup=warmup, iterations=iterations)
    throughput = (batch_size * seq_len) / (result.mean / 1000)

    return {"mean_ms": result.mean, "memory_gb": result.memory_gb, "throughput": throughput}


def _compute_scaling_exponent(results: list[dict]) -> float | None:
    """Compute empirical scaling exponent O(L^x) from benchmark results."""
    if len(results) < 2:
        return None
    first, last = results[0], results[-1]
    len_ratio = last["seq_len"] / first["seq_len"]
    time_ratio = last["mean_ms"] / first["mean_ms"]
    if len_ratio <= 1 or time_ratio <= 0:
        return None
    return math.log(time_ratio) / math.log(len_ratio)


def run_scaling_benchmark(
    seq_lengths: list[int] | None = None,
    iterations: int = 20,
):
    """Run scaling benchmark across sequence lengths.

    Benchmarks KSSM and a CausalTransformer baseline at the same model scale,
    then prints a comparison table and scaling exponents.
    """
    if seq_lengths is None:
        seq_lengths = [2048, 4096, 8192, 16384, 32768, 65536]
    batch_size, d_model, n_layers, n_heads = 1, 768, 12, 12
    d_ff = d_model * 4  # 3072

    print("=" * 70)
    print("Scaling Benchmark: KSSM vs Transformer Throughput")
    print("=" * 70)
    print(f"batch_size: {batch_size}, d_model: {d_model}, n_layers: {n_layers}, n_heads: {n_heads}")

    device = torch.device("cuda")

    # Phase 1: Benchmark KSSM
    print(f"\n--- KSSM ---")
    print(f"{'SeqLen':>8} | {'tok/s':>12} | {'Time (ms)':>10} | {'Memory (GB)':>12}")
    print("-" * 50)

    kssm_results = []
    for seq_len in seq_lengths:
        gc.collect()
        torch.cuda.empty_cache()

        config = KSSMConfig(d_model=d_model, d_inner=d_model * 2, n_layers=n_layers, n_heads=32)
        model = KSSMBackbone(config)

        result = benchmark_model(model, batch_size, seq_len, d_model, device=device, iterations=iterations)
        print(f"{seq_len:>8} | {result['throughput']:>12,.0f} | {result['mean_ms']:>10.2f} | {result['memory_gb']:>12.2f}")
        kssm_results.append({"seq_len": seq_len, "model": "KSSM", **result})
        del model

    # Phase 2: Benchmark Transformer baseline
    print(f"\n--- CausalTransformer (SDPA/FlashAttention) ---")
    print(f"{'SeqLen':>8} | {'tok/s':>12} | {'Time (ms)':>10} | {'Memory (GB)':>12}")
    print("-" * 50)

    transformer_results = []
    for seq_len in seq_lengths:
        gc.collect()
        torch.cuda.empty_cache()

        try:
            model = CausalTransformer(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff)
            result = benchmark_model(model, batch_size, seq_len, d_model, device=device, iterations=iterations)
            print(f"{seq_len:>8} | {result['throughput']:>12,.0f} | {result['mean_ms']:>10.2f} | {result['memory_gb']:>12.2f}")
            transformer_results.append({"seq_len": seq_len, "model": "Transformer", **result})
            del model
        except RuntimeError:
            # OOM at long sequences is expected for O(L^2) attention
            print(f"{seq_len:>8} | {'OOM':>12} | {'OOM':>10} | {'OOM':>12}")
            transformer_results.append({"seq_len": seq_len, "model": "Transformer", "mean_ms": None, "memory_gb": None, "throughput": None})

    # Phase 3: Comparison table
    print(f"\n{'=' * 80}")
    print("Side-by-Side Comparison")
    print(f"{'=' * 80}")
    print(f"{'SeqLen':>8} | {'KSSM tok/s':>12} | {'KSSM GB':>8} | {'Trans tok/s':>12} | {'Trans GB':>9} | {'Speedup':>8}")
    print("-" * 75)

    for k, t in zip(kssm_results, transformer_results):
        seq_len = k["seq_len"]
        k_toks = f"{k['throughput']:>12,.0f}"
        k_gb = f"{k['memory_gb']:>8.2f}"

        if t["throughput"] is not None:
            t_toks = f"{t['throughput']:>12,.0f}"
            t_gb = f"{t['memory_gb']:>9.2f}"
            speedup = k["throughput"] / t["throughput"]
            sp_str = f"{speedup:>7.1f}x"
        else:
            t_toks = f"{'OOM':>12}"
            t_gb = f"{'OOM':>9}"
            sp_str = f"{'∞':>8}"

        print(f"{seq_len:>8} | {k_toks} | {k_gb} | {t_toks} | {t_gb} | {sp_str}")

    # Phase 4: Scaling analysis
    print(f"\n{'=' * 70}")
    print("Scaling Analysis")
    print(f"{'=' * 70}")

    kssm_exp = _compute_scaling_exponent(kssm_results)
    if kssm_exp is not None:
        first, last = kssm_results[0], kssm_results[-1]
        len_ratio = last["seq_len"] / first["seq_len"]
        print(f"  Seq length range: {first['seq_len']} -> {last['seq_len']} ({len_ratio:.0f}x)")
        print(f"  KSSM empirical exponent:        O(L^{kssm_exp:.2f})  (expected: O(L^1.0))")

    valid_transformer = [r for r in transformer_results if r["throughput"] is not None]
    transformer_exp = _compute_scaling_exponent(valid_transformer)
    if transformer_exp is not None:
        print(f"  Transformer empirical exponent:  O(L^{transformer_exp:.2f})  (expected: O(L^2.0))")

    # Phase 5: Mamba-2 reference
    print(f"\nMamba-2 Reference (Dao & Gu, 2024):")
    print(f"  Mamba-2 SSD claims 2-8x faster than Mamba-1 selective scan.")
    print(f"  At moderate lengths (~2K-8K), comparable to Transformer+FlashAttention.")
    print(f"  At long sequences (>16K), SSMs (Mamba-2, KSSM) expected faster due to O(L) scaling.")

    return {
        "kssm": kssm_results,
        "transformer": transformer_results,
    }


if __name__ == "__main__":
    seed_everything()
    run_scaling_benchmark()
