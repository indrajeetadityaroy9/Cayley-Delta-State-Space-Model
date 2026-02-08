"""KSSM Backbone: Stack of KSSMBlocks with optional hybrid layers."""

import torch.nn as nn
from torch import Tensor

from kssm.config.defaults import KSSMConfig, METABOLIC_LAMBDA
from .kssm_block import KSSMBlock
from .components import RMSNorm


class KSSMBackbone(nn.Module):
    """KSSM Backbone: Stack of KSSMBlocks.

    Pure homogeneous SSM architecture using State Space Duality.

    Usage (instant, no data required):
        config = KSSMConfig(d_model=512, n_layers=8)
        backbone = KSSMBackbone(config)
    """

    def __init__(self, config: KSSMConfig):
        super().__init__()
        self.config = config

        self.blocks = nn.ModuleList()
        for i in range(config.n_layers):
            self.blocks.append(KSSMBlock(config, layer_idx=i))

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
        
        This applies the standardized sparsity penalty (METABOLIC_LAMBDA)
        to the mean gate activation.

        Returns:
            Scalar tensor: Standardized auxiliary loss to be added to the task loss.
        """
        total_loss = 0.0
        for block in self.blocks:
            if hasattr(block, "_metabolic_loss"):
                total_loss += block._metabolic_loss
        
        return (total_loss / len(self.blocks)) * METABOLIC_LAMBDA
