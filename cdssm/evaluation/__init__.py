"""Evaluation package."""

from cdssm.evaluation.evaluator import evaluate_epoch
from cdssm.evaluation.metrics import perplexity_from_loss

__all__ = ["evaluate_epoch", "perplexity_from_loss"]
