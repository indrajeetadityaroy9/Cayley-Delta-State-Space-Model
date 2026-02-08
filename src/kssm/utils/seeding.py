import os
import random
import numpy as np
import torch

_INTERNAL_SEED = 42

def seed_everything(seed: int = _INTERNAL_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_data_seed(split: str) -> int:
    offsets = {"train": 1000, "validation": 2000, "test": 3000}
    return _INTERNAL_SEED + offsets.get(split, 0)

def init_worker_rng(worker_id: int) -> None:
    worker_seed = _INTERNAL_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)