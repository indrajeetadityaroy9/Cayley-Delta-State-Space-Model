"""Reproducibility primitives: seeds, determinism, device selection."""

import os
import random

import numpy as np
import torch


# Fixed internal seed for reproducibility (not exposed as argument)
_INTERNAL_SEED = 42


def seed_everything():
    """Set all random seeds for reproducibility.

    Uses a fixed internal seed to ensure deterministic results.
    This function should be called at the start of each experiment.
    """
    os.environ["PYTHONHASHSEED"] = str(_INTERNAL_SEED)
    random.seed(_INTERNAL_SEED)
    np.random.seed(_INTERNAL_SEED)
    torch.manual_seed(_INTERNAL_SEED)
    torch.cuda.manual_seed_all(_INTERNAL_SEED)


def get_data_seed(split: str) -> int:
    """Get deterministic seed for data split.

    Ensures no overlap between calibration, training, and evaluation data
    by using different seed offsets for each split.

    Args:
        split: One of 'calibration', 'train', or 'eval'

    Returns:
        Deterministic seed for the specified split
    """
    offsets = {
        'calibration': 0,
        'train': 1000,
        'eval': 2000,
    }
    if split not in offsets:
        raise ValueError(f"Unknown split: {split}. Must be one of {list(offsets.keys())}")
    return _INTERNAL_SEED + offsets[split]


def init_worker_rng(worker_id: int) -> None:
    """Initialize worker RNG for reproducible DataLoader behavior.

    Each worker gets a unique but deterministic seed based on the
    internal seed plus worker ID.

    Args:
        worker_id: DataLoader worker ID
    """
    worker_seed = _INTERNAL_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
