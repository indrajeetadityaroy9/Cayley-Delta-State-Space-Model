"""KSSM Language Model."""

import torch.nn as nn
from torch import Tensor

from kssm.config import KSSMConfig
from kssm.model.backbone import KSSMBackbone
from kssm.modules.components import compute_variance_preserving_std


class KSSMLMHeadModel(nn.Module):
    """KSSM Language Model.

    Usage (instant, no data required):
        config = KSSMConfig(d_model=512, n_layers=8)
        model = KSSMLMHeadModel(config, vocab_size=50257)

    Usage (with calibration for optimal accuracy):
        config = KSSMConfig(d_model=512, n_layers=8)
        bounds = calibrate_spectral_bounds(dataloader, config.d_model)
        config = config.with_calibration(**bounds)
        model = KSSMLMHeadModel(config, vocab_size=50257)
    """

    def __init__(self, config: KSSMConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.backbone = KSSMBackbone(config)
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
