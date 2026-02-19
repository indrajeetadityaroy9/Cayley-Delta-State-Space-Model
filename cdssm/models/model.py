"""CDSSM Language Model."""

import torch.nn as nn
from torch import Tensor

from cdssm.config.model import CDSSMConfig
from cdssm.models.block import CDSSMBlock
from cdssm.models.modules import RMSNorm
from cdssm.models.init import compute_variance_preserving_std


class CDSSMLMHeadModel(nn.Module):
    """CDSSM Language Model with tied embedding/LM-head weights.

    Embedding -> N x CDSSMBlock -> RMSNorm -> LM head (weight-tied).
    Passes v_first (value residual) across layers for RWKV-7 style interpolation.
    """

    def __init__(self, config: CDSSMConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.blocks = nn.ModuleList(
            CDSSMBlock(config, layer_idx=i) for i in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        stds = compute_variance_preserving_std(
            config.d_model, config.d_inner, config.n_layers
        )
        nn.init.normal_(self.embedding.weight, std=stds["embedding"])

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass through embedding, backbone, and LM head.

        Args:
            input_ids: (batch, seq_len) - input token IDs

        Returns:
            logits: (batch, seq_len, vocab_size) - output logits
        """
        x = self.embedding(input_ids)
        v_first = None
        for block in self.blocks:
            x, v_first = block(x, v_first=v_first)
        x = self.final_norm(x)
        return self.lm_head(x)
