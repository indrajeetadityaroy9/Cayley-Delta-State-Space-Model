"""Selective Copying: Input Gating Validation.

Data tokens (1-15) interspersed with noise (0). Query triggers reproduction.
Tests content-dependent gating and long-range memory across sequence lengths.

SOTA reference: Competitive SSMs achieve >95% at L=4096.
"""

import torch

from experiments.seed import seed_everything
from experiments.training import run_synthetic_experiment


NOISE_TOKEN = 0


def generate_selective_copying_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int = 17,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate selective copying batch. Returns (tokens, targets, mask)."""
    n_data_tokens = 16
    query_token = vocab_size - 1

    tokens = torch.full((batch_size, seq_len), NOISE_TOKEN, device=device, dtype=torch.long)
    targets = torch.zeros_like(tokens)
    mask = torch.zeros(batch_size, seq_len, device=device)

    for b in range(batch_size):
        data = torch.randint(1, vocab_size - 1, (n_data_tokens,), device=device)
        available_positions = seq_len - n_data_tokens - 1
        positions = torch.randperm(available_positions, device=device)[:n_data_tokens]
        positions = positions.sort().values

        for i, pos in enumerate(positions):
            tokens[b, pos] = data[i]

        query_pos = seq_len - n_data_tokens - 1
        tokens[b, query_pos] = query_token

        for i in range(n_data_tokens):
            target_pos = query_pos + 1 + i
            targets[b, target_pos] = data[i]
            mask[b, target_pos] = 1.0

    return tokens, targets, mask


def run_selective_copying(
    n_steps: int = 2000,
    seq_lengths: list[int] | None = None,
):
    """Run selective copying across multiple sequence lengths.

    Sweeps sequence length to measure how accuracy degrades with distance,
    demonstrating the model's long-range content-dependent gating ability.
    """
    if seq_lengths is None:
        seq_lengths = [256, 512, 1024, 2048, 4096]
    vocab_size = 17

    print("=" * 60)
    print("Selective Copying: Sequence Length Sweep")
    print("=" * 60)
    print(f"Vocab: 0=noise, 1-15=data, 16=query")

    results = {}
    for seq_len in seq_lengths:
        result = run_synthetic_experiment(
            experiment_name=f"Selective Copying (L={seq_len})",
            data_generator=generate_selective_copying_batch,
            vocab_size=vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=8,
            n_steps=n_steps,
            task_type='selective_copying',
            batch_size=32,
            seq_len=seq_len,
        )
        results[seq_len] = result

    # Summary table
    print(f"\n{'=' * 60}")
    print("Selective Copying Summary")
    print(f"{'=' * 60}")
    print(f"{'Seq Len':>8} | {'Accuracy':>10} | {'Above Random':>14}")
    print("-" * 40)
    for sl, r in results.items():
        print(f"{sl:>8} | {r['accuracy']:>10.1%} | {r['above_random']:>14.1%}")

    print(f"\nSOTA Reference: >95% at L=4096 (Mamba, HGRN2)")

    return results


if __name__ == "__main__":
    seed_everything()
    run_selective_copying()
