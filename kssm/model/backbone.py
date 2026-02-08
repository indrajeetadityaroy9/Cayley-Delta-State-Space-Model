"""KSSM Backbone: Stack of KSSMBlocks with optional hybrid layers."""

import torch.nn as nn
from torch import Tensor

from kssm.config import KSSMConfig
from kssm.modules.kssm_block import KSSMBlock
from kssm.modules.components import RMSNorm


class KSSMBackbone(nn.Module):
    """KSSM Backbone: Stack of KSSMBlocks.

    Pure homogeneous SSM architecture using State Space Duality.

    Usage (instant, no data required):
        config = KSSMConfig(d_model=512, n_layers=8)
        backbone = KSSMBackbone(config)

    Usage (with calibration):
        config = KSSMConfig(d_model=512, n_layers=8)
        bounds = calibrate_spectral_bounds(dataloader, config.d_model)
        config = config.with_calibration(**bounds)
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
