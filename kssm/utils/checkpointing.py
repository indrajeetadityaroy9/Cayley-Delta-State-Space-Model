"""Gradient checkpointing utilities for KSSM.

Provides memory-efficient training by trading compute for memory:
- Recomputes forward activations during backward pass
- Reduces memory usage by ~70-80% for long sequences

Usage:
    # In backbone.py, wrap each block:
    if self.use_checkpointing and self.training:
        x = checkpoint_kssm_block(block, x)
    else:
        x = block(x)
"""

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def checkpoint_kssm_block(
    block: torch.nn.Module,
    x: Tensor,
    use_reentrant: bool = False,
) -> Tensor:
    """Apply gradient checkpointing to a KSSM block.

    Wraps the block forward pass in torch.utils.checkpoint.checkpoint,
    which saves memory by not storing intermediate activations during
    forward pass. Instead, they are recomputed during backward.

    Args:
        block: The KSSMBlock module to checkpoint.
        x: Input tensor, shape (batch, seq, d_model).
        use_reentrant: Whether to use reentrant checkpointing.
                       False (default) is recommended for PyTorch 2.x.

    Returns:
        Output tensor, shape (batch, seq, d_model).

    Note:
        - use_reentrant=False is more memory efficient and supports
          arbitrary autograd graph structures.
        - use_reentrant=True is legacy mode, may be slightly faster
          but has limitations with complex graphs.
    """
    return checkpoint(
        block,
        x,
        use_reentrant=use_reentrant,
    )


def checkpoint_sequential(
    blocks: torch.nn.ModuleList,
    x: Tensor,
    segments: int = 1,
    use_reentrant: bool = False,
) -> Tensor:
    """Apply gradient checkpointing to sequential blocks.

    Divides the blocks into segments and checkpoints each segment.
    More segments = more memory savings but more recomputation.

    Args:
        blocks: ModuleList of KSSMBlock modules.
        x: Input tensor, shape (batch, seq, d_model).
        segments: Number of checkpoint segments.
                  1 = checkpoint every block (max memory savings).
                  len(blocks) = no checkpointing.
        use_reentrant: Whether to use reentrant checkpointing.

    Returns:
        Output tensor, shape (batch, seq, d_model).
    """
    if segments >= len(blocks):
        # No checkpointing needed
        for block in blocks:
            x = block(x)
        return x

    # Divide blocks into segments
    segment_size = (len(blocks) + segments - 1) // segments

    for i in range(0, len(blocks), segment_size):
        segment_blocks = blocks[i : i + segment_size]

        def run_segment(x, blocks=segment_blocks):
            for block in blocks:
                x = block(x)
            return x

        x = checkpoint(
            run_segment,
            x,
            use_reentrant=use_reentrant,
        )

    return x


def estimate_memory_savings(
    n_layers: int,
    batch_size: int,
    seq_len: int,
    d_model: int,
    d_inner: int,
    checkpointing: bool = True,
) -> dict:
    """Estimate memory savings from gradient checkpointing.

    Args:
        n_layers: Number of KSSM layers.
        batch_size: Batch size.
        seq_len: Sequence length.
        d_model: Model dimension.
        d_inner: Inner SSM dimension.
        checkpointing: Whether checkpointing is enabled.

    Returns:
        dict with estimated memory usage.
    """
    # Bytes per element (bf16)
    element_size = 2

    # Per-layer activation sizes (approximate)
    # - KSSM states: (batch, seq, d_inner, 2)
    # - A_bar: (batch, seq, d_inner, 4)
    # - u_bar: (batch, seq, d_inner, 2)
    # - MLP hidden: (batch, seq, d_model * expand)
    kssm_activations = batch_size * seq_len * d_inner * (2 + 4 + 2) * element_size
    mlp_activations = batch_size * seq_len * d_model * 2 * element_size  # expand=2

    per_layer_activations = kssm_activations + mlp_activations
    total_activations_bytes = per_layer_activations * n_layers

    if checkpointing:
        # With checkpointing, only keep 1 layer's activations at a time
        # Plus the final output for backward
        checkpoint_activations = per_layer_activations * 2
        savings_pct = (1 - checkpoint_activations / total_activations_bytes) * 100
    else:
        checkpoint_activations = total_activations_bytes
        savings_pct = 0

    return {
        "without_checkpoint_mb": total_activations_bytes / (1024 * 1024),
        "with_checkpoint_mb": checkpoint_activations / (1024 * 1024),
        "savings_pct": savings_pct,
        "per_layer_mb": per_layer_activations / (1024 * 1024),
    }
