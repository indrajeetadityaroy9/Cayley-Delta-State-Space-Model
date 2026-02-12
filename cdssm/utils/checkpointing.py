import datetime
from pathlib import Path
import torch
import torch.nn as nn
from cdssm.config import CDSSMConfig

def save_checkpoint(
    model: nn.Module,
    config: CDSSMConfig,
    step: int,
    tokens_seen: int,
    path: str | Path,
    optimizer: torch.optim.Optimizer,
    metrics: dict,
    seed: int
) -> Path:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "seed": seed,
        "timestamp": datetime.datetime.now().isoformat(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path
