"""Inference package."""

from cdssm.inference.predict import load_model_from_checkpoint, generate_greedy

__all__ = ["load_model_from_checkpoint", "generate_greedy"]
