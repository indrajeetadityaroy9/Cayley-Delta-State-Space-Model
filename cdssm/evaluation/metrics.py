"""Evaluation metrics."""

import math


def perplexity_from_loss(loss: float, vocab_size: int) -> float:
    """Compute numerically-stable perplexity from mean token cross entropy."""
    return math.exp(min(loss, math.log(vocab_size)))
