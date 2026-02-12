"""CDSSM Backbone: Stack of CDSSMBlocks with optional hybrid layers."""

import torch.nn as nn
from torch import Tensor

from cdssm.config import CDSSMConfig, derived_metabolic_lambda
from cdssm.models.block import CDSSMBlock
from cdssm.models.components import RMSNorm


class CDSSMBackbone(nn.Module):
    """CDSSM Backbone: Stack of CDSSMBlocks.

    Pure homogeneous SSM architecture using State Space Duality.

    Usage (instant, no data required):
        config = CDSSMConfig(d_model=512, n_layers=8)
        backbone = CDSSMBackbone(config)
    """

    def __init__(self, config: CDSSMConfig):
        super().__init__()
        self.config = config

        self.blocks = nn.ModuleList()
        for i in range(config.n_layers):
            self.blocks.append(CDSSMBlock(config, layer_idx=i))

        self.final_norm = RMSNorm(config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all blocks.

        Args:
            x: (batch, seq_len, d_model) - input embeddings

        Returns:
            output: (batch, seq_len, d_model) - output features
        """
        for block in self.blocks:
            x = block(x)

        return self.final_norm(x)

    def get_metabolic_loss(self) -> Tensor:
        """Aggregate auxiliary utility loss from all blocks.

        Sparsity penalty λ = 1/log(vocab_size)³ is derived from vocab size,
        providing scale-invariant pressure independent of dataset or model size.

        Returns:
            Scalar tensor: Derived auxiliary loss to be added to the task loss.
        """
        total_loss = 0.0
        for block in self.blocks:
            total_loss += block._metabolic_loss

        lam = derived_metabolic_lambda(self.config.vocab_size)
        return (total_loss / len(self.blocks)) * lam
