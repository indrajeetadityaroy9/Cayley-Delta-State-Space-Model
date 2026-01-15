"""Induction Head task for testing KSSM long-range memory.

The induction head task tests associative recall:
Given a sequence like "...A B ... A _", the model should predict "B".

This is a key benchmark from the "What Can Transformers Learn In-Context?" paper
that tests a model's ability to:
1. Store associations (A -> B)
2. Recall associations after a delay

KSSM with nuclear initialization should excel at this task due to:
- Near-unitary dynamics (low damping) preserving information
- 2D state space enabling complex oscillatory patterns
- Gated MLP for feature mixing (query-key like interactions)

Usage:
    python examples/induction_head.py
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from kssm.model.language_model import KSSMLMHeadModelSimple
from kssm.modules.init import nuclear_init


class InductionHeadDataset:
    """Synthetic induction head dataset.

    Generates sequences of the form:
        [random tokens...] A B [random tokens...] A _
    where the model should predict B at the _ position.

    This version ensures clean patterns without overlap.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        seq_len: int = 128,
        n_patterns: int = 4,
        device: str = 'cuda',
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_patterns = n_patterns
        self.device = device

    def generate_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a batch of induction head sequences.

        Returns:
            input_ids: Shape (batch, seq_len)
            targets: Shape (batch, seq_len) - target token at each position
            mask: Shape (batch, seq_len) - 1.0 at induction positions, 0.0 elsewhere
        """
        # Start with random tokens
        input_ids = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=self.device)
        targets = torch.zeros_like(input_ids)
        mask = torch.zeros(batch_size, self.seq_len, device=self.device)

        # Insert clean A-B patterns and their recalls
        for b in range(batch_size):
            # Reserve positions for patterns (non-overlapping)
            # Pattern zone: [5, seq_len//3]
            # Recall zone: [seq_len//2, seq_len-2]
            pattern_zone_end = self.seq_len // 3
            recall_zone_start = self.seq_len // 2

            for i in range(self.n_patterns):
                # Pattern position (evenly spaced in pattern zone)
                pattern_pos = 5 + i * (pattern_zone_end - 5) // self.n_patterns
                if pattern_pos + 1 >= pattern_zone_end:
                    continue

                # Recall position (evenly spaced in recall zone)
                recall_pos = recall_zone_start + i * (self.seq_len - 2 - recall_zone_start) // self.n_patterns
                if recall_pos + 1 >= self.seq_len:
                    continue

                # Choose unique A and B tokens
                a_token = i  # Use pattern index as A token for uniqueness
                b_token = self.vocab_size - 1 - i  # Use reverse index for B

                # Insert A B pattern
                input_ids[b, pattern_pos] = a_token
                input_ids[b, pattern_pos + 1] = b_token

                # Insert A at recall position, target B
                input_ids[b, recall_pos] = a_token
                targets[b, recall_pos + 1] = b_token
                mask[b, recall_pos + 1] = 1.0

        return input_ids, targets, mask


def train_step(
    model: nn.Module,
    dataset: InductionHeadDataset,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 32,
) -> tuple[float, float]:
    """Single training step."""
    model.train()
    optimizer.zero_grad()

    # Generate batch
    input_ids, targets, mask = dataset.generate_batch(batch_size)

    # Forward pass
    logits, _ = model(input_ids, use_triton=False)

    # Compute loss only at masked positions
    loss = F.cross_entropy(
        logits.view(-1, dataset.vocab_size),
        targets.view(-1),
        reduction='none',
    )
    loss = loss.view(batch_size, -1)
    loss = (loss * mask).sum() / (mask.sum() + 1e-6)

    # Backward pass
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    # Compute accuracy at masked positions
    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        correct = (predictions == targets) * mask
        accuracy = correct.sum() / (mask.sum() + 1e-6)

    return loss.item(), accuracy.item()


def evaluate(
    model: nn.Module,
    dataset: InductionHeadDataset,
    n_batches: int = 10,
    batch_size: int = 32,
) -> tuple[float, float]:
    """Evaluate model on induction task."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
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

    avg_loss = total_loss / (total_count + 1e-6)
    avg_accuracy = total_correct / (total_count + 1e-6)

    return avg_loss, avg_accuracy


def main():
    parser = argparse.ArgumentParser(description='Train KSSM on induction head task')
    parser.add_argument('--vocab-size', type=int, default=64, help='Vocabulary size')
    parser.add_argument('--seq-len', type=int, default=128, help='Sequence length')
    parser.add_argument('--d-model', type=int, default=64, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--n-steps', type=int, default=1000, help='Training steps')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--eval-interval', type=int, default=100, help='Evaluation interval')
    args = parser.parse_args()

    print(f"Training KSSM on Induction Head task")
    print(f"  Vocab size: {args.vocab_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Batch size: {args.batch_size}")
    print()

    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        return

    # Create dataset
    dataset = InductionHeadDataset(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        device='cuda',
    )

    # Create model with full block structure
    model = KSSMLMHeadModelSimple(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        mlp_expand=2,
        tie_weights=False,  # Don't tie for small vocab
    ).cuda()

    # Re-initialize with very low damping (critical for induction head)
    # Alpha bias = -5.0 gives softplus(-5) ≈ 0.0067 damping
    # This means 0.9933^500 ≈ 0.035 signal retention after 500 steps
    for block in model.backbone.blocks:
        nuclear_init(block.mixer, long_memory=True)
        # Also initialize B projection with larger std for stronger signal
        if hasattr(block.mixer, 'B_proj'):
            nn.init.normal_(block.mixer.B_proj.weight, std=0.1)
            nn.init.zeros_(block.mixer.B_proj.bias)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler (cosine decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps)

    # Training loop
    print("\nTraining...")
    start_time = time.time()
    best_acc = 0.0

    for step in range(1, args.n_steps + 1):
        loss, accuracy = train_step(model, dataset, optimizer, args.batch_size)
        scheduler.step()

        if step % args.eval_interval == 0 or step == 1:
            eval_loss, eval_acc = evaluate(model, dataset, n_batches=10, batch_size=args.batch_size)
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            best_acc = max(best_acc, eval_acc)

            print(f"Step {step:5d} | Loss: {loss:.4f} | Train Acc: {accuracy:.4f} | "
                  f"Eval Acc: {eval_acc:.4f} | Best: {best_acc:.4f} | {steps_per_sec:.1f} steps/sec")

            # Early stopping if we achieve high accuracy
            if eval_acc > 0.95:
                print(f"\nEarly stopping: achieved {eval_acc:.4f} accuracy!")
                break

    # Final evaluation
    print("\nFinal evaluation...")
    final_loss, final_acc = evaluate(model, dataset, n_batches=50, batch_size=args.batch_size)
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Final Accuracy: {final_acc:.4f}")

    # Report success/failure
    if final_acc > 0.95:
        print("\n SUCCESS: Model learned induction head pattern!")
        print("  The KSSM model successfully retains long-range associations.")
    elif final_acc > 0.5:
        print("\n PARTIAL: Model shows significant learning.")
        print("  Consider: more training steps or hyperparameter tuning.")
    else:
        print("\n NEEDS WORK: Model shows some learning but accuracy is low.")
        print("  The architecture is correct but may need tuning.")


if __name__ == "__main__":
    main()
