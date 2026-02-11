#!/usr/bin/env python
import argparse
import math
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from kssm.config.defaults import KSSMConfig
from kssm.models.language_model import KSSMLMHeadModel
from kssm.data.datasets import WikiTextDataset
from kssm.training.trainer import train_one_epoch
from kssm.training.optim import build_param_groups, build_cosine_schedule
from kssm.evaluation.evaluator import evaluate_epoch
from kssm.utils.seeding import seed_everything, get_data_seed, init_worker_rng
from kssm.utils.checkpointing import save_checkpoint

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config_dict = load_config(args.config)
    seed = args.seed if args.seed is not None else config_dict.get("seed", 42)
    seed_everything(seed)

    device = torch.device(config_dict["training"]["device"])

    # Data
    context_length = config_dict["data"]["context_length"]
    train_dataset = WikiTextDataset("train", context_length)
    val_dataset = WikiTextDataset("validation", context_length)

    vocab_size = train_dataset.tokenizer.vocab_size
    n_layers = config_dict["model"]["n_layers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_dict["training"]["batch_size"],
        shuffle=True,
        num_workers=config_dict["data"]["num_workers"],
        pin_memory=True,
        worker_init_fn=init_worker_rng,
        generator=torch.Generator().manual_seed(get_data_seed("train")),
        persistent_workers=True,
        prefetch_factor=config_dict["data"]["prefetch_factor"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config_dict["training"]["batch_size"],
        shuffle=False,
        num_workers=config_dict["data"]["num_workers"],
        pin_memory=True,
        worker_init_fn=init_worker_rng,
        persistent_workers=True,
        prefetch_factor=config_dict["data"]["prefetch_factor"],
    )

    # Model (vocab_size now part of config for derived constants)
    model_config = KSSMConfig(
        d_model=config_dict["model"]["d_model"],
        n_layers=n_layers,
        context_length=context_length,
        vocab_size=vocab_size,
    )
    model = KSSMLMHeadModel(model_config, vocab_size).to(device)
    if config_dict["training"]["precision"] == "bfloat16":
        model = model.bfloat16()

    model = torch.compile(model, mode="reduce-overhead")

    # Optimizer (SSM LR ratio derived from n_layers)
    param_groups = build_param_groups(model, base_lr=float(config_dict["training"]["base_lr"]), n_layers=n_layers)
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config_dict["training"]["weight_decay"],
        betas=tuple(config_dict["training"]["betas"]),
        fused=True
    )

    grad_accum_steps = config_dict["training"]["grad_accum_steps"]
    epochs = config_dict["training"]["epochs"]
    total_steps = len(train_loader) * epochs // grad_accum_steps
    scheduler = build_cosine_schedule(optimizer, config_dict["training"]["warmup_steps"], total_steps)

    best_val_ppl = float("inf")
    checkpoint_dir = Path(config_dict["checkpointing"]["save_dir"])

    # PPL clamp derived from vocab size
    ppl_clamp = math.log(vocab_size)

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, grad_accum_steps, vocab_size=vocab_size)
        val_loss = evaluate_epoch(model, val_loader, device)

        train_ppl = math.exp(min(train_loss, ppl_clamp))
        val_ppl = math.exp(min(val_loss, ppl_clamp))

        print(f"Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            save_checkpoint(
                model=model,
                config=model_config,
                step=epoch * len(train_loader) // grad_accum_steps,
                tokens_seen=epoch * len(train_dataset) * context_length,
                path=checkpoint_dir / f"{config_dict['experiment_name']}_best.pt",
                optimizer=optimizer,
                metrics={"best_val_ppl": best_val_ppl, "epoch": epoch},
                seed=seed
            )
            print(f"Saved best model (PPL: {best_val_ppl:.2f})")

if __name__ == "__main__":
    main()
