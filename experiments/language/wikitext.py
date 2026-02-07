"""WikiText Language Modeling Benchmark.

Trains KSSM on WikiText-103 and measures perplexity.
"""

import datetime
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from experiments.metrics import compute_loss
from experiments.seed import _INTERNAL_SEED, get_data_seed, init_worker_rng, seed_everything
from experiments.training import build_cosine_schedule, build_param_groups
from kssm.config import KSSMConfig
from kssm.model.language_model import KSSMLMHeadModel
from kssm.modules.calibration import calibrate_spectral_bounds


def save_checkpoint(
    model: nn.Module,
    config: Any,
    step: int,
    tokens_seen: int,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    metrics: dict | None = None,
) -> Path:
    """Save checkpoint in canonical format."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "seed": _INTERNAL_SEED,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if metrics is not None:
        checkpoint["metrics"] = metrics

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def load_checkpoint(path: str | Path, device: str = "cuda") -> dict:
    """Load checkpoint and validate canonical format."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required_keys = ["model_state_dict", "config", "step", "tokens_seen"]
    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint missing required keys: {missing}")
    return checkpoint


class WikiTextDataset(Dataset):
    """WikiText dataset tokenized with GPT-2 tokenizer."""

    def __init__(self, split: str = "train", context_length: int = 1024, dataset_name: str = "wikitext-103"):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_length = context_length

        # Map short names to full HuggingFace dataset names
        dataset_map = {
            "wikitext-103": "wikitext-103-raw-v1",
            "wikitext-2": "wikitext-2-raw-v1",
        }
        hf_dataset = dataset_map.get(dataset_name, dataset_name)

        print(f"Loading {dataset_name} {split} split...")
        dataset = load_dataset("wikitext", hf_dataset, split=split)

        print("Tokenizing...")
        texts = [ex["text"] for ex in dataset if ex["text"].strip()]
        tokenized = self.tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
        all_tokens = [tok for ids in tokenized["input_ids"] for tok in ids]

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"Total tokens: {len(self.tokens):,}")

        self.n_sequences = (len(self.tokens) - 1) // context_length
        print(f"Sequences: {self.n_sequences:,}")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length
        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]
        return x, y


def train_one_epoch(model, dataloader, optimizer, scheduler, device, grad_accum_steps=8):
    model.train()
    total_loss = 0.0
    total_batches = 0  # Explicit counter instead of batch_idx + 1
    step = 0

    optimizer.zero_grad()
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = compute_loss(logits, y)
        loss = loss / grad_accum_steps
        loss.backward()

        total_loss += loss.item() * grad_accum_steps
        total_batches += 1

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % 100 == 0:
                avg_loss = total_loss / total_batches
                ppl = math.exp(min(avg_loss, 20.0))
                elapsed = time.time() - start_time
                tok_per_sec = total_batches * x.shape[0] * x.shape[1] / elapsed
                print(f"  Step {step:5d} | Loss {avg_loss:.4f} | PPL {ppl:.2f} | {tok_per_sec:.0f} tok/s")

    # Handle final incomplete accumulation batch - don't lose gradient signal
    if total_batches % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / total_batches if total_batches > 0 else 0.0


@torch.no_grad()
def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = compute_loss(logits, y)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_wikitext(
    dataset: str = "wikitext-103",
    epochs: int = 20,
    context_length: int = 1024,
    resume_from: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
):
    """Run WikiText language modeling experiment.

    Args:
        dataset: Dataset name ("wikitext-103" or "wikitext-2")
        epochs: Number of training epochs
        context_length: Sequence length for training
        resume_from: Optional path to checkpoint to resume from
        checkpoint_dir: Directory for saving checkpoints (default: <project_root>/checkpoints)
    """
    if checkpoint_dir is None:
        checkpoint_dir = _PROJECT_ROOT / "checkpoints"
    checkpoint_dir = Path(checkpoint_dir)
    model_name = "KSSMLMHeadModel"

    d_model, n_layers = 768, 12
    batch_size, grad_accum_steps = 4, 8

    print("=" * 60)
    print(f"WikiText-103 Language Modeling | Model: {model_name}")
    print("=" * 60)
    print(f"Model: d_model={d_model}, n_layers={n_layers}")
    print(f"Context: {context_length}, Batch: {batch_size} x {grad_accum_steps}")

    device = torch.device("cuda")

    train_dataset = WikiTextDataset("train", context_length, dataset_name=dataset)
    val_dataset = WikiTextDataset("validation", context_length, dataset_name=dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        worker_init_fn=init_worker_rng,
        generator=torch.Generator().manual_seed(get_data_seed('train')),
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
        worker_init_fn=init_worker_rng,
        persistent_workers=True,
        prefetch_factor=4,
    )

    vocab_size = train_dataset.tokenizer.vocab_size

    # Calibrate spectral bounds from training data subset with isolated calibration seed.
    # This ensures calibration data is disjoint from validation/test per the documented
    # split isolation methodology (calibration=+0, train=+1000, eval=+2000).
    print("\nCalibrating spectral bounds from training data (calibration split)...")
    calib_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        worker_init_fn=init_worker_rng,
        generator=torch.Generator().manual_seed(get_data_seed('calibration')),
        prefetch_factor=4,
    )
    calib_embedding = torch.nn.Embedding(vocab_size, d_model).to(device)
    bounds = calibrate_spectral_bounds(calib_loader, d_model, n_batches=20, embedding=calib_embedding)
    del calib_embedding, calib_loader
    print(f"Calibrated: t=[{bounds['t_min']:.2f}, {bounds['t_max']:.2f}], freq=[{bounds['freq_min']:.4f}, {bounds['freq_max']:.4f}]")

    config = KSSMConfig(
        d_model=d_model,
        d_inner=d_model * 2,
        n_layers=n_layers,
        n_heads=32,
    ).with_calibration(**bounds)

    model = KSSMLMHeadModel(config, vocab_size).to(device).bfloat16()
    model = torch.compile(model, mode="reduce-overhead")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    param_groups = build_param_groups(model, base_lr=6e-4, ssm_lr_factor=0.1)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.1, betas=(0.9, 0.95), fused=True)
    total_steps = len(train_loader) * epochs // grad_accum_steps
    scheduler = build_cosine_schedule(optimizer, 500, total_steps)

    start_epoch = 1
    best_val_ppl = float("inf")

    # Resume from checkpoint if provided
    if resume_from is not None:
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = load_checkpoint(resume_from, device=str(device))
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "metrics" in checkpoint and "epoch" in checkpoint["metrics"]:
            start_epoch = checkpoint["metrics"]["epoch"] + 1
        if "metrics" in checkpoint and "best_val_ppl" in checkpoint["metrics"]:
            best_val_ppl = checkpoint["metrics"]["best_val_ppl"]
        print(f"Resuming from epoch {start_epoch}, best_val_ppl={best_val_ppl:.2f}")

    best_epoch = start_epoch
    final_val_ppl = float("inf")

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, grad_accum_steps)
        val_loss = evaluate_epoch(model, val_loader, device)

        train_ppl = math.exp(min(train_loss, 20.0))
        val_ppl = math.exp(min(val_loss, 20.0))
        final_val_ppl = val_ppl  # Track final epoch's PPL

        print(f"Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_epoch = epoch
            save_checkpoint(
                model=model,
                config=config,
                step=epoch * len(train_loader) // grad_accum_steps,
                tokens_seen=epoch * len(train_dataset) * context_length,
                path=checkpoint_dir / "wikitext_best.pt",
                optimizer=optimizer,
                metrics={"best_val_ppl": best_val_ppl, "epoch": epoch},
            )
            print(f"Saved best model (PPL: {best_val_ppl:.2f})")

    # Report both best and final metrics for transparency
    print(f"\n{'='*50}")
    print(f"Final Results ({model_name}):")
    print(f"  Best Val PPL:  {best_val_ppl:.2f} (epoch {best_epoch})")
    print(f"  Final Val PPL: {final_val_ppl:.2f} (epoch {epochs})")
    print(f"{'='*50}")

    print(f"\n{'='*50}")
    print("Reference Baselines (WikiText-103, ~44M params):")
    print(f"  Transformer (GPT-style):  ~24.40 PPL")
    print(f"  Mamba (S6):               ~22.58 PPL")
    print(f"  HGRN2:                    ~23.10 PPL")
    print(f"  KSSM (this run):          {best_val_ppl:.2f} PPL")
    print(f"{'='*50}")

    return {
        "model": model_name,
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
        "final_val_ppl": final_val_ppl,
    }


if __name__ == "__main__":
    seed_everything()
    run_wikitext()
