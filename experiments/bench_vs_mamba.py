"""Throughput benchmark: KSSM vs Mamba.

Compares throughput between KSSM and Mamba SSM implementations.
Target: KSSM should be within 1.5x of Mamba's throughput.

Usage:
    python experiments/bench_vs_mamba.py [--batch-size 4] [--d-model 768]
"""

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent to path for kssm imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kssm.config import KSSMConfig
from kssm.model.backbone import KSSMBackbone


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_kssm_model(d_model: int, n_layers: int, d_state: int = 16, use_checkpointing: bool = True) -> nn.Module:
    """Create KSSM backbone model."""
    config = KSSMConfig(
        d_model=d_model,
        d_inner=d_model * 2,  # Standard 2x expansion
        d_state=d_state,
        n_layers=n_layers,
        use_checkpointing=use_checkpointing,
    )
    return KSSMBackbone(config)


def create_mamba_model(d_model: int, n_layers: int, d_state: int = 16, expand: int = 2) -> nn.Module:
    """Create Mamba backbone model."""
    from mamba_ssm import Mamba

    class MambaBackbone(nn.Module):
        def __init__(self, d_model, n_layers, d_state, expand):
            super().__init__()
            self.blocks = nn.ModuleList([
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=4,
                    expand=expand,
                )
                for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.norm(x)

    return MambaBackbone(d_model, n_layers, d_state, expand)


def benchmark_model(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_warmup: int = 5,
    n_iter: int = 20,
    backward: bool = True,
) -> dict:
    """Benchmark a model's forward and backward pass."""
    model = model.cuda().bfloat16()
    model.train()

    x = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(n_warmup):
        if backward:
            x_in = x.clone().requires_grad_(True)
            out = model(x_in)
            if isinstance(out, tuple):
                out = out[0]
            loss = out.sum()
            loss.backward()
        else:
            with torch.no_grad():
                out = model(x)

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Time forward pass
    start_fwd = torch.cuda.Event(enable_timing=True)
    end_fwd = torch.cuda.Event(enable_timing=True)

    start_fwd.record()
    for _ in range(n_iter):
        with torch.no_grad():
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
    end_fwd.record()
    torch.cuda.synchronize()
    fwd_ms = start_fwd.elapsed_time(end_fwd) / n_iter

    # Time forward + backward
    if backward:
        start_total = torch.cuda.Event(enable_timing=True)
        end_total = torch.cuda.Event(enable_timing=True)

        torch.cuda.reset_peak_memory_stats()
        start_total.record()
        for _ in range(n_iter):
            x_in = x.clone().requires_grad_(True)
            out = model(x_in)
            if isinstance(out, tuple):
                out = out[0]
            loss = out.sum()
            loss.backward()
        end_total.record()
        torch.cuda.synchronize()
        total_ms = start_total.elapsed_time(end_total) / n_iter
        bwd_ms = total_ms - fwd_ms
    else:
        total_ms = fwd_ms
        bwd_ms = 0.0

    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    tokens_per_iter = batch_size * seq_len
    throughput = tokens_per_iter / (total_ms / 1000)

    return {
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "total_ms": total_ms,
        "peak_memory_gb": peak_memory_gb,
        "throughput_tok_s": throughput,
    }


def run_benchmark(
    batch_size: int = 4,
    d_model: int = 768,
    n_layers: int = 12,
    d_state: int = 16,
    seq_lengths: list = None,
    output_dir: Path = None,
):
    """Run the KSSM vs Mamba benchmark."""
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]

    print("=" * 80)
    print("KSSM vs Mamba: Throughput Comparison")
    print("=" * 80)
    print(f"Config: batch_size={batch_size}, d_model={d_model}, n_layers={n_layers}, d_state={d_state}")
    print()

    # Create models and count parameters
    kssm_model = create_kssm_model(d_model, n_layers, d_state)
    mamba_model = create_mamba_model(d_model, n_layers, d_state)

    kssm_params = count_parameters(kssm_model)
    mamba_params = count_parameters(mamba_model)

    print(f"KSSM parameters:  {kssm_params:,}")
    print(f"Mamba parameters: {mamba_params:,}")
    print(f"Parameter ratio: {mamba_params / kssm_params:.2f}x")
    print()

    del kssm_model, mamba_model
    gc.collect()
    torch.cuda.empty_cache()

    results = []

    header = f"{'Seq Len':>10} | {'KSSM (tok/s)':>15} | {'Mamba (tok/s)':>15} | {'KSSM Mem':>10} | {'Mamba Mem':>10} | {'Ratio':>10}"
    print(header)
    print("-" * len(header))

    for seq_len in seq_lengths:
        # Benchmark KSSM
        gc.collect()
        torch.cuda.empty_cache()

        try:
            kssm_model = create_kssm_model(d_model, n_layers, d_state, use_checkpointing=True)
            kssm_model.gradient_checkpointing_enable()
            kssm_result = benchmark_model(kssm_model, batch_size, seq_len, d_model)
            del kssm_model
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                kssm_result = {"throughput_tok_s": 0, "peak_memory_gb": float("inf"), "oom": True}
            else:
                raise

        gc.collect()
        torch.cuda.empty_cache()

        # Benchmark Mamba
        try:
            mamba_model = create_mamba_model(d_model, n_layers, d_state)
            mamba_result = benchmark_model(mamba_model, batch_size, seq_len, d_model)
            del mamba_model
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                mamba_result = {"throughput_tok_s": 0, "peak_memory_gb": float("inf"), "oom": True}
            else:
                raise

        gc.collect()
        torch.cuda.empty_cache()

        # Calculate ratio
        if mamba_result.get("oom") or mamba_result["throughput_tok_s"] == 0:
            ratio = "OOM"
            mamba_throughput = "OOM"
            mamba_mem = "OOM"
        else:
            ratio = f"{kssm_result['throughput_tok_s'] / mamba_result['throughput_tok_s']:.2f}x"
            mamba_throughput = f"{mamba_result['throughput_tok_s']:,.0f}"
            mamba_mem = f"{mamba_result['peak_memory_gb']:.2f} GB"

        if kssm_result.get("oom"):
            kssm_throughput = "OOM"
            kssm_mem = "OOM"
        else:
            kssm_throughput = f"{kssm_result['throughput_tok_s']:,.0f}"
            kssm_mem = f"{kssm_result['peak_memory_gb']:.2f} GB"

        print(f"{seq_len:>10} | {kssm_throughput:>15} | {mamba_throughput:>15} | {kssm_mem:>10} | {mamba_mem:>10} | {ratio:>10}")

        results.append({
            "seq_len": seq_len,
            "kssm_throughput": kssm_result.get("throughput_tok_s", 0),
            "kssm_memory_gb": kssm_result.get("peak_memory_gb", float("inf")),
            "kssm_fwd_ms": kssm_result.get("fwd_ms", 0),
            "kssm_bwd_ms": kssm_result.get("bwd_ms", 0),
            "kssm_oom": kssm_result.get("oom", False),
            "mamba_throughput": mamba_result.get("throughput_tok_s", 0),
            "mamba_memory_gb": mamba_result.get("peak_memory_gb", float("inf")),
            "mamba_fwd_ms": mamba_result.get("fwd_ms", 0),
            "mamba_bwd_ms": mamba_result.get("bwd_ms", 0),
            "mamba_oom": mamba_result.get("oom", False),
        })

    print("=" * 80)

    # Summary
    avg_ratio = sum(
        r["kssm_throughput"] / r["mamba_throughput"]
        for r in results
        if r["mamba_throughput"] > 0 and r["kssm_throughput"] > 0
    ) / len([r for r in results if r["mamba_throughput"] > 0 and r["kssm_throughput"] > 0])

    print(f"\nAverage KSSM/Mamba throughput ratio: {avg_ratio:.2f}x")
    if avg_ratio >= 0.67:  # Within 1.5x
        print("PASS: KSSM is within 1.5x of Mamba throughput")
    else:
        print("NOTE: KSSM is slower than 1.5x of Mamba (expected for less-optimized kernels)")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "kssm_vs_mamba.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {csv_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="KSSM vs Mamba throughput benchmark")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--d-model", type=int, default=768, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--d-state", type=int, default=16, help="State dimension")
    parser.add_argument("--output", type=str, default="experiments/results", help="Output directory")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=None,
                        help="Sequence lengths to test")

    args = parser.parse_args()

    run_benchmark(
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        seq_lengths=args.seq_lengths,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
