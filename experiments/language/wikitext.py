"""WikiText-103 language modeling benchmark (canonical deterministic path)."""

import datetime
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from kssm.config import KSSMConfig
from kssm.model.language_model import KSSMLMHeadModel
from kssm.modules.calibration import calibrate_spectral_bounds


_INTERNAL_SEED = 42
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def seed_everything() -> None:
    os.environ["PYTHONHASHSEED"] = str(_INTERNAL_SEED)
    random.seed(_INTERNAL_SEED)
    np.random.seed(_INTERNAL_SEED)
    torch.manual_seed(_INTERNAL_SEED)
    torch.cuda.manual_seed_all(_INTERNAL_SEED)


def get_data_seed(split: str) -> int:
    offsets = {"calibration": 0, "train": 1000, "eval": 2000}
    return _INTERNAL_SEED + offsets[split]


def init_worker_rng(worker_id: int) -> None:
    worker_seed = _INTERNAL_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)


def build_param_groups(model: nn.Module, base_lr: float) -> list[dict]:
    ssm_keywords = [
        "dynamics_proj",
        "adaptive_dt",
        "log_dt_scale",
        "decay_gate_logit",
        "decay_gate_proj",
        "selection_B",
        "selection_C",
        "selection_dt",
    ]
    ssm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ssm_keywords):
            ssm_params.append(param)
        else:
            other_params.append(param)
    return [
        {"params": other_params, "lr": base_lr},
        {"params": ssm_params, "lr": base_lr * 0.1},
    ]


def build_cosine_schedule(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: nn.Module,
    config: KSSMConfig,
    step: int,
    tokens_seen: int,
    path: str | Path,
    optimizer: torch.optim.Optimizer,
    metrics: dict,
) -> Path:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "seed": _INTERNAL_SEED,
        "timestamp": datetime.datetime.now().isoformat(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


class WikiTextDataset(Dataset):
    """WikiText-103 tokenized with GPT-2 tokenizer."""

    def __init__(self, split: str, context_length: int):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_length = context_length

        print(f"Loading wikitext-103 {split} split...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

        print("Tokenizing...")
        texts = [ex["text"] for ex in dataset if ex["text"].strip()]
        tokenized = self.tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
        all_tokens = [tok for ids in tokenized["input_ids"] for tok in ids]

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.n_sequences = (len(self.tokens) - 1) // context_length

        print(f"Total tokens: {len(self.tokens):,}")
        print(f"Sequences: {self.n_sequences:,}")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


def train_one_epoch(model, dataloader, optimizer, scheduler, device, grad_accum_steps=8):
    model.train()
    total_loss = 0.0
    total_batches = 0
    step = 0

    optimizer.zero_grad()
    start_time = time.time()
    accum_since_step = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = compute_loss(logits, y)
        (loss / grad_accum_steps).backward()
        accum_since_step += 1

        total_loss += loss.item()
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

    if accum_since_step % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / total_batches


@torch.no_grad()
def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += compute_loss(logits, y).item()
        n_batches += 1

    return total_loss / n_batches


def run_wikitext(epochs: int = 20, context_length: int = 1024):
    checkpoint_dir = _PROJECT_ROOT / "checkpoints"
    model_name = "KSSMLMHeadModel"

    d_model, n_layers = 768, 12
    batch_size, grad_accum_steps = 4, 8

    print("=" * 60)
    print(f"WikiText-103 Language Modeling | Model: {model_name}")
    print("=" * 60)
    print(f"Model: d_model={d_model}, n_layers={n_layers}")
    print(f"Context: {context_length}, Batch: {batch_size} x {grad_accum_steps}")

    device = torch.device("cuda")

    train_dataset = WikiTextDataset("train", context_length)
    val_dataset = WikiTextDataset("validation", context_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=init_worker_rng,
        generator=torch.Generator().manual_seed(get_data_seed("train")),
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=init_worker_rng,
        persistent_workers=True,
        prefetch_factor=4,
    )

    vocab_size = train_dataset.tokenizer.vocab_size

    print("\nCalibrating spectral bounds from training data...")
    calib_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=init_worker_rng,
        generator=torch.Generator().manual_seed(get_data_seed("calibration")),
        prefetch_factor=4,
    )
    calib_embedding = torch.nn.Embedding(vocab_size, d_model).to(device)
    bounds = calibrate_spectral_bounds(calib_loader, d_model, n_batches=20, embedding=calib_embedding)
    print(
        f"Calibrated: t=[{bounds['t_min']:.2f}, {bounds['t_max']:.2f}], "
        f"freq=[{bounds['freq_min']:.4f}, {bounds['freq_max']:.4f}]"
    )

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

    param_groups = build_param_groups(model, base_lr=6e-4)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.1, betas=(0.9, 0.95), fused=True)
    total_steps = len(train_loader) * epochs // grad_accum_steps
    scheduler = build_cosine_schedule(optimizer, 500, total_steps)

    best_val_ppl = float("inf")
    best_epoch = 1
    final_val_ppl = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, grad_accum_steps)
        val_loss = evaluate_epoch(model, val_loader, device)

        train_ppl = math.exp(min(train_loss, 20.0))
        val_ppl = math.exp(min(val_loss, 20.0))
        final_val_ppl = val_ppl

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

    print(f"\n{'='*50}")
    print(f"Final Results ({model_name}):")
    print(f"  Best Val PPL:  {best_val_ppl:.2f} (epoch {best_epoch})")
    print(f"  Final Val PPL: {final_val_ppl:.2f} (epoch {epochs})")
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
