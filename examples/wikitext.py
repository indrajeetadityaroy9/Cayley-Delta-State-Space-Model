"""WikiText-2 Language Modeling Benchmark for KSSM.

This is the critical validation: proving KSSM can model real language distributions.

Success Criteria:
1. Train loss drops smoothly below 4.0
2. Validation PPL comparable to Transformer baseline (~20-30 on WT2)
3. No loss spikes (KSSM's A-stability should prevent this)

Usage:
    pip install datasets transformers
    python examples/wikitext.py
"""

import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Optional: wandb for logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from kssm.config import KSSMConfig
from kssm.model.language_model import KSSMLMHeadModel
from kssm.modules.init import init_kssm_model


class WikiTextDataset(Dataset):
    """WikiText dataset for causal language modeling."""

    def __init__(
        self,
        split: str = "train",
        context_length: int = 1024,
        tokenizer_name: str = "gpt2",
    ):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_length = context_length

        # Load dataset
        print(f"Loading WikiText-2 {split} split...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # Tokenize all text
        print("Tokenizing...")
        all_tokens = []
        for example in dataset:
            text = example["text"]
            if text.strip():  # Skip empty lines
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(tokens)

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"Total tokens: {len(self.tokens):,}")

        # Calculate number of complete sequences
        self.n_sequences = len(self.tokens) // context_length
        print(f"Number of sequences: {self.n_sequences:,}")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length
        x = self.tokens[start:end]
        # Target is shifted by 1
        y = self.tokens[start + 1:end + 1]
        return x, y


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine learning rate schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    return math.exp(min(loss, 20))  # Cap to avoid overflow


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_triton: bool = False,
    log_interval: int = 10,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    step = 0
    grad_norms = []

    optimizer.zero_grad()
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        logits, _ = model(x, use_triton=use_triton)

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100,
        )

        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps
        loss.backward()

        total_loss += loss.item() * grad_accum_steps
        total_tokens += y.numel()

        # Gradient accumulation step
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Compute gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            grad_norms.append(grad_norm.item())

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            # Logging
            if step % log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                ppl = compute_perplexity(avg_loss)
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / elapsed
                lr = scheduler.get_last_lr()[0]

                print(f"  Step {step:5d} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                      f"Grad: {grad_norm:.4f} | LR: {lr:.2e} | {tokens_per_sec:.0f} tok/s")

    avg_loss = total_loss / len(dataloader)
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

    return {
        "loss": avg_loss,
        "ppl": compute_perplexity(avg_loss),
        "grad_norm": avg_grad_norm,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_triton: bool = False,
) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x, use_triton=use_triton)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100,
            reduction='sum',
        )

        total_loss += loss.item()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    return {
        "loss": avg_loss,
        "ppl": compute_perplexity(avg_loss),
    }


def main():
    parser = argparse.ArgumentParser(description='Train KSSM on WikiText-2')

    # Model config
    parser.add_argument('--d-model', type=int, default=768, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--d-inner', type=int, default=16, help='Inner SSM dimension')
    parser.add_argument('--mlp-expand', type=int, default=2, help='MLP expansion factor')

    # Training config
    parser.add_argument('--context-length', type=int, default=1024, help='Context length')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--grad-accum', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Max gradient norm')

    # Other
    parser.add_argument('--use-triton', action='store_true', help='Use Triton kernels')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create config
    config = KSSMConfig(
        d_model=args.d_model,
        d_inner=args.d_inner,
        n_layers=args.n_layers,
        mlp_expand=args.mlp_expand,
        expand_factor=1,  # d_inner is set explicitly
    )

    print(f"\nModel Config:")
    print(f"  d_model: {config.d_model}")
    print(f"  d_inner: {config.d_inner}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  mlp_expand: {config.mlp_expand}")
    print()

    # Load datasets
    try:
        train_dataset = WikiTextDataset("train", args.context_length)
        val_dataset = WikiTextDataset("validation", args.context_length)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nPlease install required packages:")
        print("  pip install datasets transformers")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Create model
    vocab_size = train_dataset.tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    model = KSSMLMHeadModel(config, vocab_size).to(device)

    # Re-initialize with nuclear init for long memory
    init_kssm_model(model, long_memory=True)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")

    # Apply torch.compile for non-Triton parts (MLP, LayerNorm)
    # Note: Use "default" mode, not "reduce-overhead" which uses CUDA graphs
    # that conflict with Triton kernels
    if args.compile:
        print("Applying torch.compile (default mode)...")
        model = torch.compile(model, mode="default")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\nTraining Config:")
    print(f"  Context length: {args.context_length}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print()

    # Wandb logging
    if args.wandb and HAS_WANDB:
        wandb.init(
            project="kssm-wikitext",
            config=vars(args),
        )

    # Training loop
    best_val_ppl = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"=== Epoch {epoch}/{args.epochs} ===")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_accum_steps=args.grad_accum,
            max_grad_norm=args.max_grad_norm,
            use_triton=args.use_triton,
            log_interval=args.log_interval,
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, use_triton=args.use_triton)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Train PPL: {train_metrics['ppl']:.2f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val PPL: {val_metrics['ppl']:.2f}")
        print(f"  Avg Grad Norm: {train_metrics['grad_norm']:.4f}")
        print()

        # Save best model
        if val_metrics['ppl'] < best_val_ppl:
            best_val_ppl = val_metrics['ppl']
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ppl': best_val_ppl,
                'config': config,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  Saved best model (PPL: {best_val_ppl:.2f})")

        # Wandb logging
        if args.wandb and HAS_WANDB:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_metrics['loss'],
                "train_ppl": train_metrics['ppl'],
                "val_loss": val_metrics['loss'],
                "val_ppl": val_metrics['ppl'],
                "grad_norm": train_metrics['grad_norm'],
                "lr": scheduler.get_last_lr()[0],
            })

    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best Validation PPL: {best_val_ppl:.2f}")
    print(f"{'='*50}")

    # Final assessment
    if best_val_ppl < 30:
        print("\n SUCCESS: KSSM achieves competitive perplexity!")
        print("  Architecture is validated for language modeling.")
    elif best_val_ppl < 50:
        print("\n PARTIAL: KSSM shows learning but PPL is high.")
        print("  May need more training or hyperparameter tuning.")
    else:
        print("\n NEEDS WORK: PPL is too high.")
        print("  Check: model capacity, learning rate, or architecture.")


if __name__ == "__main__":
    main()
