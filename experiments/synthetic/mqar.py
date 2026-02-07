"""MQAR: Multi-Query Associative Recall.

Tests associative memory capacity by sweeping number of key-value pairs.
Uses canonical training pipeline from experiments.training for consistency.

SOTA reference (Mamba, HGRN2): >95% at 4 pairs, >80% at 64 pairs (L=512).
"""

import torch

from experiments.seed import seed_everything
from experiments.training import run_synthetic_experiment


VOCAB_SIZE = 256


def generate_mqar_batch(batch_size: int, seq_len: int, num_pairs: int, device: str = "cuda"):
    """Generate MQAR batch with key-value pairs and queries.

    Returns (tokens, targets, mask=None) where targets uses -100 for ignored positions.
    """
    pairs_len = num_pairs * 2
    padding_len = seq_len - pairs_len - num_pairs

    inputs, targets = [], []
    for _ in range(batch_size):
        keys = torch.randperm(VOCAB_SIZE, device=device)[:num_pairs]
        values = torch.randint(0, VOCAB_SIZE, (num_pairs,), device=device)
        pairs = torch.stack([keys, values], dim=1).reshape(-1)
        padding = torch.zeros(padding_len, dtype=torch.long, device=device)

        order = torch.randperm(num_pairs, device=device)
        queries, answers = keys[order], values[order]

        inputs.append(torch.cat([pairs, padding, queries]))
        ignore = torch.full((pairs_len + padding_len,), -100, dtype=torch.long, device=device)
        targets.append(torch.cat([ignore, answers]))

    return torch.stack(inputs), torch.stack(targets), None


def run_mqar(n_steps: int = 3000, seq_len: int = 512):
    """Run MQAR capacity sweep over increasing number of key-value pairs.

    Sweeps num_pairs to find the model's associative memory capacity limit.
    Reports accuracy vs. num_pairs curve for comparison against SOTA.
    """
    pair_counts = [4, 8, 16, 32, 64]

    print("=" * 60)
    print(f"MQAR Capacity Sweep | seq_len={seq_len}, vocab={VOCAB_SIZE}")
    print("=" * 60)
    print(f"Random baseline: {1.0 / VOCAB_SIZE:.4f}")

    results = {}
    for num_pairs in pair_counts:
        # Ensure sequence is long enough for pairs + padding + queries
        min_len = num_pairs * 3 + 10
        effective_seq_len = max(seq_len, min_len)

        result = run_synthetic_experiment(
            experiment_name=f"MQAR (pairs={num_pairs}, L={effective_seq_len})",
            data_generator=generate_mqar_batch,
            vocab_size=VOCAB_SIZE,
            d_model=128,
            n_layers=4,
            n_heads=8,
            n_steps=n_steps,
            task_type='mqar',
            batch_size=32,
            seq_len=effective_seq_len,
            num_pairs=num_pairs,
        )
        results[num_pairs] = result

    # Summary table
    print(f"\n{'=' * 60}")
    print("MQAR Capacity Summary")
    print(f"{'=' * 60}")
    print(f"{'Pairs':>8} | {'Accuracy':>10} | {'Above Random':>14} | {'Rel. Improvement':>18}")
    print("-" * 60)
    for np, r in results.items():
        print(f"{np:>8} | {r['accuracy']:>10.1%} | {r['above_random']:>14.1%} | {r['relative_improvement']:>17.1f}x")

    print(f"\nSOTA Reference (Mamba/HGRN2, ~same scale):")
    print(f"  4 pairs:  >95% | 64 pairs: >80%")

    return results


if __name__ == "__main__":
    seed_everything()
    run_mqar()
