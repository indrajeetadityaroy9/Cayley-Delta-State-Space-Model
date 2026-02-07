"""Loss, accuracy, baselines, and evaluation result types."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class EvaluationResult:
    """Evaluation result with baseline-relative metrics.

    Reports raw accuracy plus comparison to computed baselines,
    avoiding arbitrary pass/fail thresholds.
    """
    accuracy: float
    random_baseline: float

    @property
    def above_random(self) -> float:
        """How much accuracy exceeds random baseline (0-1 scale above random)."""
        if self.random_baseline >= 1.0:
            return 0.0
        return (self.accuracy - self.random_baseline) / (1.0 - self.random_baseline)

    @property
    def relative_improvement(self) -> float:
        """Relative improvement over random baseline (as multiplier)."""
        if self.random_baseline <= 0:
            return float('inf') if self.accuracy > 0 else 1.0
        return self.accuracy / self.random_baseline


def compute_random_baseline(task_type: str, **task_params) -> float:
    """Compute theoretical random baseline for a task.

    Args:
        task_type: One of 'mqar', 'selective_copying', 'needle', 'multi_choice', 'lm'
        **task_params: Task-specific parameters (vocab_size, n_choices, etc.)

    Returns:
        Random baseline accuracy (0-1)
    """
    if task_type == 'mqar':
        vocab_size = task_params.get('vocab_size', 256)
        return 1.0 / vocab_size

    elif task_type == 'selective_copying':
        vocab_size = task_params.get('vocab_size', 17)
        n_data_tokens = vocab_size - 2
        return 1.0 / n_data_tokens

    elif task_type == 'needle':
        return 1.0 / 16

    elif task_type == 'induction':
        vocab_size = task_params.get('vocab_size', 64)
        return 1.0 / vocab_size

    else:
        raise ValueError(f"Unknown task type: {task_type}")


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """Canonical loss computation for all experiments.

    Uses manual masking if mask is provided, otherwise uses ignore_index=-100.
    This ensures consistent gradient flow across all experiments.

    Args:
        logits: Model output (batch, seq, vocab_size)
        targets: Ground truth labels (batch, seq)
        mask: Optional mask tensor (batch, seq), 1 for valid positions
        vocab_size: Override vocab size (defaults to logits last dim)

    Returns:
        Scalar loss tensor
    """
    if vocab_size is None:
        vocab_size = logits.size(-1)

    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    if mask is not None:
        loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        loss = loss.view(targets.shape)
        mask_sum = mask.sum().clamp(min=1.0)
        return (loss * mask).sum() / mask_sum
    else:
        return F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[int, int]:
    """Compute accuracy counts for batch.

    Args:
        logits: Model output (batch, seq, vocab_size)
        targets: Ground truth labels (batch, seq)
        mask: Optional mask tensor (batch, seq), 1 for valid positions

    Returns:
        (correct_count, total_count) as integers
    """
    preds = logits.argmax(dim=-1)

    if mask is not None:
        correct = ((preds == targets) * mask).sum().item()
        total = mask.sum().item()
    else:
        valid_mask = (targets != -100)
        correct = ((preds == targets) & valid_mask).sum().item()
        total = valid_mask.sum().item()

    return int(correct), int(total)
