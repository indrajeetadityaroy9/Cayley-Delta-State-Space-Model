"""KSSM Configuration: Parameter-Free & Hardware-Aligned.

All internal constants are derived from the minimal config surface:
(d_model, n_layers, context_length, vocab_size). No magic numbers.
"""

import math
from dataclasses import dataclass, field, asdict


# === Derived Constants (replace hardcoded magic numbers) ===

def derived_gating_range(context_length: int) -> float:
    """c = log(context_length). Ties gating dynamic range to sequence length.

    Controls |λ_eff|² = |λ|^{2·c·r}. Longer contexts need stronger flushing.
    context_length=1024 → c≈6.9, context_length=8192 → c≈9.0.
    """
    return math.log(context_length)


def derived_metabolic_lambda(vocab_size: int) -> float:
    """λ = 1/log(vocab_size)³. Scale-invariant sparsity pressure.

    Initial CE ≈ log(vocab_size). Mean gate ≈ 0.5. This gives ~0.04%
    of initial task loss as gentle pressure. GPT-2 (50257) → 0.00079.
    """
    return 1.0 / math.log(vocab_size) ** 3


def derived_ssm_lr_ratio(n_layers: int) -> float:
    """SSM LR = base_lr / sqrt(2·n_layers). T-Fixup aligned.

    n_layers=12 → 0.204x, n_layers=24 → 0.144x, n_layers=48 → 0.102x.
    Clamped to 0.5 for very shallow models.
    """
    return min(0.5, 1.0 / math.sqrt(2 * n_layers))


@dataclass
class KSSMConfig:
    """KSSM Configuration.

    Strictly parameter-free architecture derived from `d_model`, `n_layers`,
    `context_length`, and `vocab_size`. All internal constants (gating range,
    sparsity penalty, LR ratio, EMA decay) are derived from these values.

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
    vocab_size: int = 50257     # Set by training script from tokenizer

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
