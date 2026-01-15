"""KSSM Language Model: Full LM with embeddings and output head.

This provides a complete language model wrapper around the KSSM backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from kssm.config import KSSMConfig
from kssm.model.backbone import KSSMBackbone, KSSMBackboneSimple
from kssm.modules.init import init_kssm_model


class KSSMLMHeadModel(nn.Module):
    """KSSM Language Model with LM Head.

    Architecture:
        Input IDs -> Embedding -> Backbone -> LM Head -> Logits

    Args:
        config: KSSMConfig with model hyperparameters.
        vocab_size: Vocabulary size.
        tie_weights: Whether to tie embedding and LM head weights.
    """

    def __init__(
        self,
        config: KSSMConfig,
        vocab_size: int,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, config.d_model)

        # KSSM backbone
        self.backbone = KSSMBackbone(config)

        # LM head
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.embedding.weight

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.embedding.weight, std=0.02)
        if not hasattr(self.lm_head, 'weight') or self.lm_head.weight is not self.embedding.weight:
            nn.init.normal_(self.lm_head.weight, std=0.02)

        # Initialize KSSM layers with nuclear init
        init_kssm_model(self, long_memory=True)

    def forward(
        self,
        input_ids: Tensor,
        states: list[Tensor] | None = None,
        use_triton: bool = True,
    ) -> tuple[Tensor, list[Tensor]]:
        """Forward pass.

        Args:
            input_ids: Input token IDs, shape (batch, seq).
            states: Optional list of initial states for each block.
            use_triton: Whether to use Triton kernels.

        Returns:
            logits: Output logits, shape (batch, seq, vocab_size).
            final_states: List of final states for each block.
        """
        # Embed tokens
        x = self.embedding(input_ids)

        # Pass through backbone
        x, final_states = self.backbone(x, states, use_triton)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits, final_states

    @torch.no_grad()
    def step(
        self,
        input_ids: Tensor,
        states: list[Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        """Single step for autoregressive inference.

        Args:
            input_ids: Input token ID, shape (batch,) or (batch, 1).
            states: List of current states for each block.

        Returns:
            logits: Output logits, shape (batch, vocab_size).
            new_states: List of updated states.
        """
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(1)

        # Embed token
        x = self.embedding(input_ids)

        # Pass through backbone (single step)
        x, new_states = self.backbone.step(x, states)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits, new_states

    def init_states(self, batch_size: int, device: torch.device = None) -> list[Tensor]:
        """Initialize states for all blocks."""
        return self.backbone.init_states(batch_size, device)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Prompt token IDs, shape (batch, seq).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_k: If set, only sample from top-k tokens.

        Returns:
            Generated token IDs, shape (batch, seq + max_new_tokens).
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Initialize states
        states = self.init_states(batch_size, device)

        # Process prompt
        for t in range(input_ids.shape[1]):
            logits, states = self.step(input_ids[:, t], states)

        # Generate new tokens
        generated = [input_ids]

        for _ in range(max_new_tokens):
            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token)

            # Step with new token
            logits, states = self.step(next_token.squeeze(1), states)

        return torch.cat(generated, dim=1)


class KSSMLMHeadModelSimple(nn.Module):
    """Simplified KSSM LM for testing.

    Uses KSSMBackboneSimple (no input expansion).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        mlp_expand: int = 2,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Backbone
        self.backbone = KSSMBackboneSimple(d_model, n_layers, mlp_expand)

        # LM head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.embedding.weight

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Initialize KSSM layers with nuclear init (very low damping)
        from kssm.modules.init import nuclear_init
        for block in self.backbone.blocks:
            nuclear_init(block.mixer, long_memory=True)

    def forward(
        self,
        input_ids: Tensor,
        states: list[Tensor] | None = None,
        use_triton: bool = True,
    ) -> tuple[Tensor, list[Tensor]]:
        """Forward pass."""
        x = self.embedding(input_ids)
        x, final_states = self.backbone(x, states, use_triton)
        logits = self.lm_head(x)
        return logits, final_states

    @torch.no_grad()
    def step(self, input_ids: Tensor, states: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
        """Single step for autoregressive inference."""
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(1)
        x = self.embedding(input_ids)
        x, new_states = self.backbone.step(x, states)
        logits = self.lm_head(x)
        return logits, new_states

    def init_states(self, batch_size: int, device: torch.device = None) -> list[Tensor]:
        """Initialize states for all blocks."""
        return self.backbone.init_states(batch_size, device)
