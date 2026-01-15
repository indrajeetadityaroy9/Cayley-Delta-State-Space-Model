"""Throughput benchmark: KSSM vs FlashAttention Transformer.

Demonstrates KSSM's O(L) linear scaling vs Transformer's O(L^2) quadratic scaling.

Usage:
    python experiments/bench_vs_transformer.py [--batch-size 4] [--d-model 768]
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


def create_kssm_model(d_model: int, n_layers: int, use_checkpointing: bool = True, compile_mlp: bool = False) -> nn.Module:
    """Create KSSM backbone model.

    Args:
        d_model: Model dimension.
        n_layers: Number of layers.
        use_checkpointing: Enable gradient checkpointing (saves memory).
        compile_mlp: Use torch.compile on MLP (faster but slow warmup).
    """
    config = KSSMConfig(
        d_model=d_model,
        d_inner=d_model * 2,  # Standard 2x expansion
        n_layers=n_layers,
        use_checkpointing=use_checkpointing,
        compile_mlp=compile_mlp,
    )
    return KSSMBackbone(config)


def create_transformer_model(d_model: int, n_layers: int, n_heads: int) -> nn.Module:
    """Create Transformer model with FlashAttention."""
    from flash_attn.modules.mha import MHA
    from flash_attn.modules.mlp import Mlp

    class FlashTransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn = MHA(
                embed_dim=d_model,
                num_heads=n_heads,
                use_flash_attn=True,
                causal=True,
            )
            self.norm2 = nn.LayerNorm(d_model)
            self.mlp = Mlp(
                in_features=d_model,
                hidden_features=d_model * 4,  # Standard 4x expansion
                activation=nn.GELU(),
            )

        def forward(self, x):
            x = x + self.attn(self.norm1(x))[0]
            x = x + self.mlp(self.norm2(x))
            return x

    class FlashTransformer(nn.Module):
        def __init__(self, d_model, n_layers, n_heads):
            super().__init__()
            self.blocks = nn.ModuleList([
                FlashTransformerBlock(d_model, n_heads)
                for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.norm(x)

    return FlashTransformer(d_model, n_layers, n_heads)


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

    # Create input
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

    # Memory usage
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Throughput (tokens per second)
    tokens_per_iter = batch_size * seq_len
    throughput = tokens_per_iter / (total_ms / 1000)

    return {
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "total_ms": total_ms,
        "peak_memory_gb": peak_memory_gb,
        "throughput_tok_s": throughput,
    }


def run_scaling_benchmark(
    batch_size: int = 4,
    d_model: int = 768,
    n_layers: int = 12,
    n_heads: int = 12,
    seq_lengths: list = None,
    output_dir: Path = None,
):
    """Run the scaling benchmark across sequence lengths."""
    if seq_lengths is None:
        seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]

    print("=" * 80)
    print("KSSM vs FlashAttention Transformer: Throughput Scaling Benchmark")
    print("=" * 80)
    print(f"Config: batch_size={batch_size}, d_model={d_model}, n_layers={n_layers}")
    print()

    # Create models and count parameters
    kssm_model = create_kssm_model(d_model, n_layers)
    transformer_model = create_transformer_model(d_model, n_layers, n_heads)

    kssm_params = count_parameters(kssm_model)
    transformer_params = count_parameters(transformer_model)

    print(f"KSSM parameters:       {kssm_params:,}")
    print(f"Transformer parameters: {transformer_params:,}")
    print(f"Parameter ratio: {transformer_params / kssm_params:.2f}x")
    print()

    results = []

    header = f"{'Seq Len':>10} | {'KSSM (tok/s)':>15} | {'Trans (tok/s)':>15} | {'KSSM Mem':>10} | {'Trans Mem':>10} | {'Speedup':>10}"
    print(header)
    print("-" * len(header))

    for seq_len in seq_lengths:
        # Benchmark KSSM
        gc.collect()
        torch.cuda.empty_cache()

        try:
            kssm_model = create_kssm_model(d_model, n_layers, use_checkpointing=True)
            # Enable checkpointing and ensure train mode
            kssm_model.gradient_checkpointing_enable()
            kssm_result = benchmark_model(
                kssm_model, batch_size, seq_len, d_model
            )
            del kssm_model
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                kssm_result = {"throughput_tok_s": 0, "peak_memory_gb": float("inf"), "oom": True}
            else:
                raise

        gc.collect()
        torch.cuda.empty_cache()

        # Benchmark Transformer
        try:
            transformer_model = create_transformer_model(d_model, n_layers, n_heads)
            transformer_result = benchmark_model(
                transformer_model, batch_size, seq_len, d_model
            )
            del transformer_model
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                transformer_result = {"throughput_tok_s": 0, "peak_memory_gb": float("inf"), "oom": True}
            else:
                raise

        gc.collect()
        torch.cuda.empty_cache()

        # Calculate speedup
        if transformer_result.get("oom") or transformer_result["throughput_tok_s"] == 0:
            speedup = "OOM"
            trans_throughput = "OOM"
            trans_mem = "OOM"
        else:
            speedup = f"{kssm_result['throughput_tok_s'] / transformer_result['throughput_tok_s']:.2f}x"
            trans_throughput = f"{transformer_result['throughput_tok_s']:,.0f}"
            trans_mem = f"{transformer_result['peak_memory_gb']:.2f} GB"

        if kssm_result.get("oom"):
            kssm_throughput = "OOM"
            kssm_mem = "OOM"
        else:
            kssm_throughput = f"{kssm_result['throughput_tok_s']:,.0f}"
            kssm_mem = f"{kssm_result['peak_memory_gb']:.2f} GB"

        print(f"{seq_len:>10} | {kssm_throughput:>15} | {trans_throughput:>15} | {kssm_mem:>10} | {trans_mem:>10} | {speedup:>10}")

        results.append({
            "seq_len": seq_len,
            "kssm_throughput": kssm_result.get("throughput_tok_s", 0),
            "kssm_memory_gb": kssm_result.get("peak_memory_gb", float("inf")),
            "kssm_fwd_ms": kssm_result.get("fwd_ms", 0),
            "kssm_bwd_ms": kssm_result.get("bwd_ms", 0),
            "kssm_oom": kssm_result.get("oom", False),
            "transformer_throughput": transformer_result.get("throughput_tok_s", 0),
            "transformer_memory_gb": transformer_result.get("peak_memory_gb", float("inf")),
            "transformer_fwd_ms": transformer_result.get("fwd_ms", 0),
            "transformer_bwd_ms": transformer_result.get("bwd_ms", 0),
            "transformer_oom": transformer_result.get("oom", False),
        })

    print("=" * 80)

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "kssm_vs_transformer.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to: {csv_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="KSSM vs Transformer throughput benchmark")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--d-model", type=int, default=768, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--n-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--output", type=str, default="experiments/results", help="Output directory")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=None,
                        help="Sequence lengths to test (default: [1024, 2048, 4096, 8192, 16384, 32768])")

    args = parser.parse_args()

    run_scaling_benchmark(
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        seq_lengths=args.seq_lengths,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
