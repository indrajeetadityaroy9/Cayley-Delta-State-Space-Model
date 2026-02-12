"""Loss functions used in training and evaluation."""

import torch
import torch.nn.functional as F


def language_model_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Token-level cross entropy with ignore index for masked targets."""
    vocab_size = logits.size(-1)
    return F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)


def total_training_loss(task_loss: torch.Tensor, auxiliary_loss: torch.Tensor) -> torch.Tensor:
    """Combine task and auxiliary regularization terms."""
    return task_loss + auxiliary_loss
