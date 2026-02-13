"""Module entrypoint for model training."""

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

_DEVICE = torch.device("cuda")

# H100 80 GB / 26 vCPU defaults
_BATCH_SIZE = 8
_NUM_WORKERS = 8
_PREFETCH_FACTOR = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    from cdssm.config.defaults import CDSSMConfig
    from cdssm.data.datasets import build_dataset
    from cdssm.evaluation.evaluator import evaluate_epoch
    from cdssm.models.model import CDSSMLMHeadModel
    from cdssm.training.optim import build_cosine_schedule, build_param_groups
    from cdssm.training.trainer import train_one_epoch
    from cdssm.utils.checkpointing import save_checkpoint
    from cdssm.utils.seeding import seed_everything

    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else config_dict.get("seed", 42)
    seed_everything(seed)

    # Data
    context_length = config_dict["data"]["context_length"]
    tokenizer_name = config_dict.get("tokenizer_name", "gpt2")
    train_dataset = build_dataset(config_dict["data"], "train", tokenizer_name=tokenizer_name)
    val_dataset = build_dataset(config_dict["data"], "validation", tokenizer_name=tokenizer_name)

    vocab_size = train_dataset.tokenizer.vocab_size
    n_layers = config_dict["model"]["n_layers"]
    ppl_clamp = math.log(vocab_size)

    # Dataloaders (inlined — no builder indirection)
    def worker_init_fn(worker_id: int) -> None:
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=_BATCH_SIZE,
        shuffle=True,
        num_workers=_NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed + 1000),
        persistent_workers=True,
        prefetch_factor=_PREFETCH_FACTOR,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=_BATCH_SIZE,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        prefetch_factor=_PREFETCH_FACTOR,
    )

    # Model
    model_config = CDSSMConfig(
        d_model=config_dict["model"]["d_model"],
        n_layers=n_layers,
        context_length=context_length,
        vocab_size=vocab_size,
        tokenizer_name=tokenizer_name,
    )
    model = CDSSMLMHeadModel(model_config).to(_DEVICE)
    if config_dict["training"].get("precision", "bfloat16") == "bfloat16":
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
    min_lr_ratio = config_dict["training"].get("min_lr_ratio", 0.0)
    scheduler = build_cosine_schedule(optimizer, config_dict["training"]["warmup_steps"], total_steps, min_lr_ratio=min_lr_ratio)

    best_val_ppl = float("inf")
    checkpoint_dir = Path(config_dict["checkpointing"]["save_dir"])

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        grad_clip = config_dict["training"].get("grad_clip", 1.0)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            grad_accum_steps,
            vocab_size=vocab_size,
            grad_clip=grad_clip,
        )
        val_loss = evaluate_epoch(model, val_loader)

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
                seed=seed,
            )
            print(f"Saved best model (PPL: {best_val_ppl:.2f})")


if __name__ == "__main__":
    main()
