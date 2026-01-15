"""KSSM Configuration."""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class KSSMConfig:
    """Configuration for KSSM layers and models.

    Attributes:
        d_model: Model dimension (input/output dimension).
        d_state: State dimension per channel. Each channel has a 2D state (pair).
        d_inner: Inner dimension after expansion. If None, defaults to d_model * expand.
        expand: Expansion factor for inner dimension (Mamba-style, default 2).
        d_conv: Convolution kernel size for local smoothing (Mamba uses 4).
        dt_rank: Rank of dt projection. If None, defaults to ceil(d_model / 16).
        n_layers: Number of KSSM layers in a backbone.
        dt_min: Minimum discretization timestep.
        dt_max: Maximum discretization timestep.
        dt_init: How to initialize dt ('random' or 'constant').
        use_checkpointing: Whether to use gradient checkpointing.
        dtype: Data type for computations (bfloat16 recommended for H100).
        device: Device for computations.
    """

    d_model: int = 768
    d_state: int = 16
    d_inner: Optional[int] = None
    expand: int = 2  # Mamba-style expansion factor
    d_conv: int = 4  # Conv kernel size for local smoothing
    dt_rank: Optional[int] = None  # Rank for dt projection
    n_layers: int = 12
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    use_checkpointing: bool = True
    dtype: torch.dtype = field(default=torch.bfloat16)
    device: str = "cuda"

    # Legacy parameters (for backward compatibility)
    expand_factor: int = 2
    mlp_expand: int = 2
    mlp_bias: bool = False
    compile_mlp: bool = False
    chunk_size: int = 1024

    def __post_init__(self):
        """Validate and compute derived values."""
        # Use expand if expand_factor not explicitly set
        if self.expand_factor != 2:
            self.expand = self.expand_factor

        if self.d_inner is None:
            self.d_inner = self.d_model * self.expand

        if self.dt_rank is None:
            # Default: ceil(d_model / 16) as in Mamba
            self.dt_rank = (self.d_model + 15) // 16

        # Ensure d_inner is even (for 2D state pairs)
        if self.d_inner % 2 != 0:
            raise ValueError(f"d_inner must be even, got {self.d_inner}")

        if self.dt_min <= 0 or self.dt_max <= 0:
            raise ValueError("dt_min and dt_max must be positive")

        if self.dt_min > self.dt_max:
            raise ValueError("dt_min must be <= dt_max")

        if self.dt_init not in ("random", "constant"):
            raise ValueError(f"dt_init must be 'random' or 'constant', got {self.dt_init}")

    @property
    def n_pairs(self) -> int:
        """Number of 2D state pairs (d_inner // 2)."""
        return self.d_inner // 2
