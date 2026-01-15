"""End-to-end benchmark: KSSM training step time vs sequence length.

This benchmark demonstrates KSSM's O(L) linear scaling in sequence length,
which is the key advantage over O(L^2) attention.

The "Money Plot" for the paper:
- X-axis: Sequence Length L
- Y-axis: Training step time (ms)
- Shows: Linear scaling with constant low overhead
"""

import argparse
import time

import torch
import torch.nn.functional as F

from kssm.config import KSSMConfig
from kssm.model.language_model import KSSMLMHeadModel
from kssm.modules.init import init_kssm_model


def benchmark_training_step(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    n_warmup: int = 5,
    n_iter: int = 20,
) -> dict:
    """Benchmark a single training step.

    Returns forward time, backward time, and total time.
    """
    model.train()

    # Create random input
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(n_warmup):
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()

    # Benchmark forward
    torch.cuda.synchronize()
    fwd_start = time.perf_counter()
    for _ in range(n_iter):
        with torch.no_grad():
            logits, _ = model(x)
    torch.cuda.synchronize()
    fwd_time = (time.perf_counter() - fwd_start) / n_iter * 1000

    # Benchmark full training step (forward + backward)
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    for _ in range(n_iter):
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - total_start) / n_iter * 1000

    bwd_time = total_time - fwd_time

    # Compute throughput
    tokens = batch_size * seq_len
    throughput = tokens / (total_time / 1000)

    return {
        "fwd_ms": fwd_time,
        "bwd_ms": bwd_time,
        "total_ms": total_time,
        "throughput": throughput,
    }


def main():
    parser = argparse.ArgumentParser(description="E2E KSSM Benchmark")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--d-model", type=int, default=768, help="Model dimension")
    parser.add_argument("--d-inner", type=int, default=16, help="Inner SSM dimension")
    parser.add_argument("--n-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--vocab-size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--n-warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--n-iter", type=int, default=20, help="Benchmark iterations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: d_model={args.d_model}, d_inner={args.d_inner}, n_layers={args.n_layers}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Create model
    config = KSSMConfig(
        d_model=args.d_model,
        d_inner=args.d_inner,
        n_layers=args.n_layers,
        mlp_expand=2,
    )
    model = KSSMLMHeadModel(config, args.vocab_size).to(device)
    init_kssm_model(model, long_memory=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print()

    # Sequence lengths to benchmark
    seq_lens = [256, 512, 1024, 2048, 4096, 8192, 16384]

    print("=" * 80)
    print(f"{'Seq Len':<10} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Total (ms)':<15} {'tok/s':<15}")
    print("-" * 80)

    results = []
    for seq_len in seq_lens:
        try:
            metrics = benchmark_training_step(
                model, args.batch_size, seq_len, args.vocab_size, device,
                n_warmup=args.n_warmup, n_iter=args.n_iter,
            )
            print(f"{seq_len:<10} {metrics['fwd_ms']:<15.2f} {metrics['bwd_ms']:<15.2f} "
                  f"{metrics['total_ms']:<15.2f} {metrics['throughput']:<15.0f}")
            results.append((seq_len, metrics))
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{seq_len:<10} OOM")
                torch.cuda.empty_cache()
            else:
                raise

    print("=" * 80)

    # Print scaling analysis
    if len(results) >= 2:
        print("\nScaling Analysis:")
        base_len, base_metrics = results[0]
        for seq_len, metrics in results[1:]:
            len_ratio = seq_len / base_len
            time_ratio = metrics['total_ms'] / base_metrics['total_ms']
            scaling = time_ratio / len_ratio
            print(f"  L={seq_len}: {len_ratio:.1f}x length -> {time_ratio:.1f}x time (scaling factor: {scaling:.2f})")

        print("\n  Ideal O(L) scaling: factor = 1.00")
        print("  O(L^2) would show factor = L_ratio")


if __name__ == "__main__":
    main()
