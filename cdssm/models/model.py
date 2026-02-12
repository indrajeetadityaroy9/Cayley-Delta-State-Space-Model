"""CDSSM Language Model."""

import torch.nn as nn
from torch import Tensor

from cdssm.config import CDSSMConfig
from cdssm.models.backbone import CDSSMBackbone
from cdssm.models.components import compute_variance_preserving_std


class CDSSMLMHeadModel(nn.Module):
    """CDSSM Language Model.

    Usage (instant, no data required):
        config = CDSSMConfig(d_model=512, n_layers=8)
        model = CDSSMLMHeadModel(config, vocab_size=50257)
        
    Self-Calibration:
        The model automatically initializes using Universal Spectral Priors
        based on the `context_length` (default 8192). No manual calibration is needed.
    """

    def __init__(self, config: CDSSMConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.backbone = CDSSMBackbone(config)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
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
        x = self.backbone(x)
        return self.lm_head(x)
