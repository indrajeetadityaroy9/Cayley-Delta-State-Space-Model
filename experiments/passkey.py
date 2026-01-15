"""Passkey Retrieval Benchmark (Needle in Haystack).

This benchmark tests KSSM's ability to preserve information over long sequences.
The task: retrieve a 5-digit passkey embedded somewhere in a long noise sequence.

Based on the working induction_head.py approach.

Usage:
    python experiments/passkey.py --seq-lengths 256 512 1024 2048
"""

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from kssm.model.language_model import KSSMLMHeadModelSimple
from kssm.modules.init import nuclear_init


# ============================================================
# Configuration
# ============================================================
VOCAB_SIZE = 64   # Same as induction head
PASSKEY_LEN = 5   # Length of passkey to remember
QUERY_TOKEN = 0   # Special token to trigger retrieval
DEVICE = "cuda"


# ============================================================
# Dataset
# ============================================================
class PasskeyDataset:
    """Passkey retrieval dataset.

    Generates sequences of the form:
        [noise] [START=1] [passkey tokens] [END=2] [noise] [QUERY=0] [passkey tokens...]

    The model must output the passkey tokens after seeing QUERY.
    Loss is computed only at passkey prediction positions.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        seq_len: int = 256,
        passkey_len: int = PASSKEY_LEN,
        device: str = DEVICE,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.passkey_len = passkey_len
        self.device = device

        # Special tokens
        self.query_token = 0
        self.start_token = 1
        self.end_token = 2
        self.min_content_token = 3  # Content tokens start at 3

    def generate_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a batch of passkey retrieval examples.

        Returns:
            input_ids: Shape (batch, seq_len)
            targets: Shape (batch, seq_len) - passkey tokens at prediction positions
            mask: Shape (batch, seq_len) - 1.0 at passkey prediction positions
        """
        # All in one sequence of length seq_len
        input_ids = torch.randint(
            self.min_content_token, self.vocab_size,
            (batch_size, self.seq_len), device=self.device
        )
        targets = torch.zeros_like(input_ids)
        mask = torch.zeros(batch_size, self.seq_len, device=self.device)

        for b in range(batch_size):
            # Generate passkey (random tokens from content range)
            passkey = torch.randint(
                self.min_content_token, self.vocab_size,
                (self.passkey_len,), device=self.device
            )

            # Layout: [noise] [START] [passkey] [END] [noise] [QUERY] [passkey for teacher forcing]
            # Passkey zone: positions 10 to 10+passkey_len+2
            passkey_start = 10

            input_ids[b, passkey_start] = self.start_token
            input_ids[b, passkey_start + 1:passkey_start + 1 + self.passkey_len] = passkey
            input_ids[b, passkey_start + 1 + self.passkey_len] = self.end_token

            # Query and prediction zone at end of sequence
            # Reserve last (passkey_len + 1) positions for QUERY + teacher forcing
            query_pos = self.seq_len - self.passkey_len - 1
            input_ids[b, query_pos] = self.query_token

            # Teacher forcing: positions after QUERY contain passkey[:-1]
            # Position query_pos+1 has passkey[0], ..., position query_pos+passkey_len-1 has passkey[-2]
            input_ids[b, query_pos + 1:query_pos + self.passkey_len] = passkey[:-1]

            # Targets: at position query_pos, predict passkey[0]; at query_pos+1, predict passkey[1]; etc.
            for i in range(self.passkey_len):
                targets[b, query_pos + i] = passkey[i]
                mask[b, query_pos + i] = 1.0

        return input_ids, targets, mask


# ============================================================
# Training
# ============================================================
def train_step(
    model: nn.Module,
    dataset: PasskeyDataset,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 16,
) -> tuple[float, float]:
    """Single training step."""
    model.train()
    optimizer.zero_grad()

    input_ids, targets, mask = dataset.generate_batch(batch_size)

    # Forward
    logits, _ = model(input_ids, use_triton=False)

    # Loss only at masked positions
    loss = F.cross_entropy(
        logits.view(-1, dataset.vocab_size),
        targets.view(-1),
        reduction='none',
    )
    loss = loss.view(batch_size, -1)
    loss = (loss * mask).sum() / (mask.sum() + 1e-6)

    # Backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Accuracy at masked positions
    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        correct = (predictions == targets) * mask
        accuracy = correct.sum() / (mask.sum() + 1e-6)

    return loss.item(), accuracy.item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: PasskeyDataset,
    n_batches: int = 10,
    batch_size: int = 16,
) -> tuple[float, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0.0

    for _ in range(n_batches):
        input_ids, targets, mask = dataset.generate_batch(batch_size)
        logits, _ = model(input_ids, use_triton=False)

        loss = F.cross_entropy(
            logits.view(-1, dataset.vocab_size),
            targets.view(-1),
            reduction='none',
        )
        loss = loss.view(batch_size, -1)
        total_loss += (loss * mask).sum().item()

        predictions = logits.argmax(dim=-1)
        correct = (predictions == targets) * mask
        total_correct += correct.sum().item()
        total_count += mask.sum().item()

    return total_loss / total_count, total_correct / total_count


# ============================================================
# Benchmark
# ============================================================
def run_passkey_benchmark(
    d_model: int = 64,
    n_layers: int = 2,
    seq_lengths: list = None,
    n_steps: int = 1000,
    batch_size: int = 16,
    output_dir: Path = None,
):
    """Run passkey retrieval benchmark across sequence lengths."""
    if seq_lengths is None:
        seq_lengths = [256, 512, 1024, 2048]

    print("=" * 80)
    print("KSSM Passkey Retrieval Benchmark (Needle in Haystack)")
    print("=" * 80)
    print(f"Config: d_model={d_model}, n_layers={n_layers}, passkey_len={PASSKEY_LEN}")
    print(f"Training: {n_steps} steps per sequence length")
    print()

    results = []

    for seq_len in seq_lengths:
        print(f"\n{'='*60}")
        print(f"Sequence Length: {seq_len}")
        print(f"{'='*60}")

        gc.collect()
        torch.cuda.empty_cache()

        try:
            # Create model (using proven architecture)
            model = KSSMLMHeadModelSimple(
                vocab_size=VOCAB_SIZE,
                d_model=d_model,
                n_layers=n_layers,
            ).to(DEVICE)

            # Apply nuclear init for long-range memory
            for block in model.backbone.blocks:
                nuclear_init(block.mixer, long_memory=True)

            # Dataset
            dataset = PasskeyDataset(
                vocab_size=VOCAB_SIZE,
                seq_len=seq_len,
                passkey_len=PASSKEY_LEN,
            )

            # Optimizer
            optimizer = optim.AdamW(model.parameters(), lr=1e-3)

            # Training
            best_acc = 0.0
            for step in range(n_steps):
                loss, acc = train_step(model, dataset, optimizer, batch_size)
                best_acc = max(best_acc, acc)

                if (step + 1) % 100 == 0 or step == 0:
                    eval_loss, eval_acc = evaluate(model, dataset, n_batches=5, batch_size=batch_size)
                    print(f"Step {step + 1:4d} | Loss: {loss:.4f} | Train: {acc*100:.1f}% | Eval: {eval_acc*100:.1f}%")

                    if eval_acc >= 0.99:
                        print(f"Converged at step {step + 1}!")
                        break

            # Final evaluation
            final_loss, final_acc = evaluate(model, dataset, n_batches=20, batch_size=batch_size)
            print(f"\nFinal Accuracy: {final_acc * 100:.1f}%")

            results.append({
                "seq_len": seq_len,
                "final_acc": final_acc,
                "best_acc": best_acc,
                "status": "success",
            })

            del model, dataset, optimizer

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at seq_len={seq_len}")
                results.append({
                    "seq_len": seq_len,
                    "final_acc": 0,
                    "best_acc": 0,
                    "status": "oom",
                })
            else:
                raise

        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Seq Length':>12} | {'Final Accuracy':>15} | {'Status':>10}")
    print("-" * 45)
    for r in results:
        if r["status"] == "success":
            print(f"{r['seq_len']:>12} | {r['final_acc']*100:>14.1f}% | {'OK':>10}")
        else:
            print(f"{r['seq_len']:>12} | {'N/A':>15} | {r['status']:>10}")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "passkey.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {csv_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="KSSM Passkey Retrieval Benchmark")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--n-steps", type=int, default=1000, help="Training steps per seq length")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=None,
                        help="Sequence lengths to test")
    parser.add_argument("--output", type=str, default="experiments/results",
                        help="Output directory")

    args = parser.parse_args()

    run_passkey_benchmark(
        d_model=args.d_model,
        n_layers=args.n_layers,
        seq_lengths=args.seq_lengths,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
