"""KSSM Configuration: Parameter-Free & Hardware-Aligned."""

from dataclasses import dataclass, field, asdict

# Robust sparsity penalty for Utility Gating.
METABOLIC_LAMBDA: float = 1e-3

@dataclass
class KSSMConfig:
    """KSSM Configuration.

    Strictly parameter-free architecture derived from `d_model` and `n_layers`.
    
    Hardware Alignment:
        - `head_dim` is fixed to 64 to exploit H100 Tensor Core vector width.
        - `d_inner` is fixed to 2 * d_model.
        - `n_heads` is derived as `d_inner // 64`.
    
    Spectral Priors:
        - Universal initialization covers timescales [1, context_length].
    """

    # === Required ===
    d_model: int = 768
    n_layers: int = 12
    context_length: int = 8192  # Defines the upper bound of the universal timescale prior

    # === Computed / Fixed (Hardware Constraints) ===
    d_inner: int = field(init=False)
    n_heads: int = field(init=False)
    head_dim: int = field(init=False, default=64)
    ssm_norm_groups: int = field(init=False, default=32)

    def __post_init__(self):
        # 1. Expand State Dimension (Standard Expansion factor = 2)
        self.d_inner = self.d_model * 2

        # 2. Hardware Alignment (Head Dim = 64 for Tensor Cores)
        if self.d_inner % 64 != 0:
            raise ValueError(
                f"d_inner ({self.d_inner}) must be divisible by head_dim=64 for Tensor Core alignment. "
                f"Adjust d_model ({self.d_model}) such that 2*d_model is divisible by 64."
            )
        
        self.n_heads = self.d_inner // 64
        
        # 3. Verify GroupNorm constraints
        if self.d_inner % 32 != 0:
             raise ValueError("d_inner must be divisible by 32 for GroupNorm.")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)