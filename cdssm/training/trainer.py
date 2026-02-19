"""Training loop for CDSSM."""

import datetime
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cdssm.config.experiment import ExperimentConfig


def compute_perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity."""
    return math.exp(loss)


@torch.no_grad()
def evaluate_epoch(model: nn.Module, dataloader) -> float:
    """Compute average cross-entropy loss over *dataloader*."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def _seed_all(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    """Manages the full training loop: epochs, gradient accumulation, checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        config,
        experiment_name: str = "default",
        grad_accum_steps: int = 8,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        seed: int = 42,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.experiment_name = experiment_name
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.seed = seed

    @classmethod
    def from_config(cls, exp: ExperimentConfig) -> "Trainer":
        """Build a fully configured Trainer from an ExperimentConfig.

        Handles seeding, dataset construction, model compilation,
        optimizer setup with depth-scaled LR, and cosine scheduling.
        """
        from cdssm.config.model import CDSSMConfig, derived_ssm_lr_ratio
        from cdssm.data.datasets import TokenDataset, build_dataloader
        from cdssm.models.model import CDSSMLMHeadModel

        seed = exp.seed
        _seed_all(seed)

        # Data
        train_dataset = TokenDataset(
            dataset_name=exp.data.dataset_name,
            dataset_config=exp.data.dataset_config,
            split="train",
            context_length=exp.data.context_length,
            num_tokens=exp.data.num_tokens,
            text_field=exp.data.text_field,
        )
        val_dataset = TokenDataset(
            dataset_name=exp.data.dataset_name,
            dataset_config=exp.data.dataset_config,
            split="validation",
            context_length=exp.data.context_length,
            num_tokens=exp.data.num_tokens,
            text_field=exp.data.text_field,
        )

        train_loader = build_dataloader(
            train_dataset, batch_size=exp.data.batch_size, shuffle=True,
            num_workers=exp.data.num_workers, seed=seed,
        )
        val_loader = build_dataloader(
            val_dataset, batch_size=exp.data.batch_size, shuffle=False,
            num_workers=exp.data.num_workers, seed=seed,
        )

        # Model
        vocab_size = train_dataset.tokenizer.vocab_size
        model_config = CDSSMConfig(
            d_model=exp.model.d_model,
            n_layers=exp.model.n_layers,
            context_length=exp.data.context_length,
            vocab_size=vocab_size,
            state_dim=exp.model.state_dim,
        )
        model = CDSSMLMHeadModel(model_config).cuda().bfloat16()
        model = torch.compile(model, mode="reduce-overhead")

        # Optimizer with depth-scaled LR for SSM dynamics parameters
        base_lr = exp.training.base_lr
        ssm_keywords = ["gate_proj", "log_dt_scale", "v_residual_gate", "shortcut_scale"]
        ssm_params = []
        other_params = []
        for name, param in model.named_parameters():
            if any(k in name for k in ssm_keywords):
                ssm_params.append(param)
            else:
                other_params.append(param)

        ssm_lr = base_lr * derived_ssm_lr_ratio(exp.model.n_layers)
        param_groups = [
            {"params": other_params, "lr": base_lr},
            {"params": ssm_params, "lr": ssm_lr},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=exp.training.weight_decay,
            betas=exp.training.betas,
            fused=True,
        )

        # Cosine schedule with warmup
        total_steps = len(train_loader) * exp.training.epochs // exp.training.grad_accum_steps
        num_warmup = exp.training.warmup_steps
        min_lr_ratio = exp.training.min_lr_ratio

        def lr_lambda(current_step):
            if current_step < num_warmup:
                return float(current_step) / float(max(1, num_warmup))
            progress = float(current_step - num_warmup) / float(max(1, total_steps - num_warmup))
            cosine_decay = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=model_config,
            experiment_name=exp.experiment_name,
            grad_accum_steps=exp.training.grad_accum_steps,
            grad_clip=exp.training.grad_clip,
            checkpoint_dir=exp.checkpoint_dir,
            seed=seed,
        )

    def fit(self, epochs: int) -> None:
        """Run full training loop."""
        best_val_ppl = float("inf")

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            print(f"\n=== Epoch {epoch}/{epochs} ===")
            train_loss = self.train_one_epoch()
            val_loss = evaluate_epoch(self.model, self.val_loader)

            train_ppl = compute_perplexity(train_loss)
            val_ppl = compute_perplexity(val_loss)

            print(f"Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "config": asdict(self.config),
                    "step": epoch * len(self.train_loader) // self.grad_accum_steps,
                    "tokens_seen": epoch * len(self.train_loader.dataset) * self.config.context_length,
                    "seed": self.seed,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metrics": {"best_val_ppl": best_val_ppl, "epoch": epoch},
                }, path)
                print(f"Saved best model (PPL: {best_val_ppl:.2f})")

        print(f"Training complete. Best Val PPL: {best_val_ppl:.2f}")

    def train_one_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        step = 0

        self.optimizer.zero_grad()
        start_time = time.time()

        log_interval = max(10, len(self.train_loader) // (self.grad_accum_steps * 10))

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

            (loss / self.grad_accum_steps).backward()
            total_loss += loss.item()
            total_batches += 1

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                step += 1

                if step % log_interval == 0:
                    avg_loss = total_loss / total_batches
                    ppl = compute_perplexity(avg_loss)
                    elapsed = time.time() - start_time
                    tok_per_sec = total_batches * x.shape[0] * x.shape[1] / elapsed
                    print(f"  Step {step:5d} | Loss {avg_loss:.4f} | PPL {ppl:.2f} | {tok_per_sec:.0f} tok/s")

        if total_batches % self.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return total_loss / total_batches
