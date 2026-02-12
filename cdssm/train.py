#!/usr/bin/env python
"""Module entrypoint for model training."""

import argparse
import os
from pathlib import Path

import torch
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    from cdssm.config import CDSSMConfig
    from cdssm.data import build_dataset, build_dataloaders
    from cdssm.evaluation.evaluator import evaluate_epoch
    from cdssm.evaluation.metrics import perplexity_from_loss
    from cdssm.models.model import CDSSMLMHeadModel
    from cdssm.training.optim import build_cosine_schedule, build_param_groups
    from cdssm.training.trainer import train_one_epoch
    from cdssm.utils.checkpointing import save_checkpoint
    from cdssm.utils.seeding import seed_everything

    config_dict = load_config(args.config)
    seed = args.seed if args.seed is not None else config_dict.get("seed", 42)
    seed_everything(seed)

    device = torch.device(config_dict["training"]["device"])

    # Data
    context_length = config_dict["data"]["context_length"]
    train_dataset = build_dataset(config_dict["data"], "train")
    val_dataset = build_dataset(config_dict["data"], "validation")

    vocab_size = train_dataset.tokenizer.vocab_size
    n_layers = config_dict["model"]["n_layers"]

    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        config_dict["data"],
        config_dict["training"],
    )

    # Model
    model_config = CDSSMConfig(
        d_model=config_dict["model"]["d_model"],
        n_layers=n_layers,
        context_length=context_length,
        vocab_size=vocab_size,
    )
    model = CDSSMLMHeadModel(model_config, vocab_size).to(device)
    if config_dict["training"]["precision"] == "bfloat16":
        model = model.bfloat16()

    model = torch.compile(model, mode="reduce-overhead")

    # Optimizer
    base_lr = float(os.environ.get("CDSSM_BASE_LR_OVERRIDE", config_dict["training"]["base_lr"]))
    param_groups = build_param_groups(model, base_lr=base_lr, n_layers=n_layers)
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config_dict["training"]["weight_decay"],
        betas=tuple(config_dict["training"]["betas"]),
        fused=True,
    )

    grad_accum_steps = config_dict["training"]["grad_accum_steps"]
    epochs = config_dict["training"]["epochs"]
    total_steps = len(train_loader) * epochs // grad_accum_steps
    scheduler = build_cosine_schedule(optimizer, config_dict["training"]["warmup_steps"], total_steps)

    best_val_ppl = float("inf")
    checkpoint_dir = Path(config_dict["checkpointing"]["save_dir"])

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            grad_accum_steps,
            vocab_size=vocab_size,
        )
        val_loss = evaluate_epoch(model, val_loader, device)

        train_ppl = perplexity_from_loss(train_loss, vocab_size)
        val_ppl = perplexity_from_loss(val_loss, vocab_size)

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
                seed=seed,
            )
            print(f"Saved best model (PPL: {best_val_ppl:.2f})")


if __name__ == "__main__":
    main()
