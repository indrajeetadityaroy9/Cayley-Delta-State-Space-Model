"""Dataloader construction utilities."""

import torch
from torch.utils.data import DataLoader

from cdssm.utils.seeding import get_data_seed, init_worker_rng


def build_dataloaders(train_dataset, val_dataset, data_cfg: dict, training_cfg: dict):
    """Build train/validation dataloaders with reproducible worker seeding."""
    batch_size = training_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    prefetch_factor = data_cfg["prefetch_factor"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=init_worker_rng,
        generator=torch.Generator().manual_seed(get_data_seed("train")),
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=init_worker_rng,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader
