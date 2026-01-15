"""KSSM Backbone: Stack of KSSM Blocks.

The backbone is the core of a KSSM model - it stacks multiple KSSMBlocks
to build deep sequence models.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from kssm.config import KSSMConfig
from kssm.modules.kssm_block import KSSMBlock, KSSMBlockSimple


class KSSMBackbone(nn.Module):
    """KSSM Backbone: A stack of KSSMBlocks.

    Architecture:
        Input -> Block_0 -> Block_1 -> ... -> Block_{n-1} -> LayerNorm -> Output

    Each block contains:
    - Pre-norm KSSM layer (time mixing)
    - Pre-norm Gated MLP (channel mixing)
    - Residual connections

    Args:
        config: KSSMConfig with model hyperparameters.
    """

    def __init__(self, config: KSSMConfig):
        super().__init__()
        self.config = config
        self.use_checkpointing = config.use_checkpointing

        # Stack of KSSM blocks
        self.blocks = nn.ModuleList([
            KSSMBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: Tensor,
        states: list[Tensor] | None = None,
        use_triton: bool = True,
    ) -> tuple[Tensor, list[Tensor]]:
        """Forward pass through all blocks.

        Args:
            x: Input tensor, shape (batch, seq, d_model).
            states: Optional list of initial states for each block.
            use_triton: Whether to use Triton kernels.

        Returns:
            output: Output tensor, shape (batch, seq, d_model).
            final_states: List of final states for each block.
        """
        if states is None:
            states = [None] * len(self.blocks)

        final_states = []

        # Use checkpointing during training when states are not tracked
        # (common case: full-sequence training without incremental state)
        use_ckpt = (
            self.use_checkpointing
            and self.training
            and all(s is None for s in states)
        )

        for block, state in zip(self.blocks, states):
            if use_ckpt:
                # Checkpoint the block forward pass
                # Note: We create a wrapper to handle the return format
                def run_block(x, block=block, use_triton=use_triton):
                    out, _ = block(x, None, use_triton)
                    return out

                x = checkpoint(run_block, x, use_reentrant=False)
                final_states.append(None)
            else:
                x, final_state = block(x, state, use_triton)
                final_states.append(final_state)

        x = self.final_norm(x)

        return x, final_states

    @torch.no_grad()
    def step(
        self,
        x: Tensor,
        states: list[Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        """Single step for autoregressive inference.

        Args:
            x: Input tensor, shape (batch, d_model).
            states: List of current states for each block.

        Returns:
            output: Output tensor, shape (batch, d_model).
            new_states: List of updated states.
        """
        new_states = []

        for block, state in zip(self.blocks, states):
            x, new_state = block.step(x, state)
            new_states.append(new_state)

        x = self.final_norm(x)

        return x, new_states

    def init_states(self, batch_size: int, device: torch.device = None) -> list[Tensor]:
        """Initialize states for all blocks.

        Args:
            batch_size: Batch size.
            device: Device for state tensors.

        Returns:
            List of zero states for each block.
        """
        return [block.init_state(batch_size, device) for block in self.blocks]

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory-efficient training.

        This should be called after model creation for training on long sequences.
        """
        self.use_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.use_checkpointing = False


class KSSMBackboneSimple(nn.Module):
    """Simplified KSSM Backbone for testing.

    Uses KSSMBlockSimple (no input expansion).
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        mlp_expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Stack of blocks
        self.blocks = nn.ModuleList([
            KSSMBlockSimple(d_model, mlp_expand, dt_min, dt_max)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        states: list[Tensor] | None = None,
        use_triton: bool = True,
    ) -> tuple[Tensor, list[Tensor]]:
        """Forward pass."""
        if states is None:
            states = [None] * len(self.blocks)

        final_states = []

        for block, state in zip(self.blocks, states):
            x, final_state = block(x, state, use_triton)
            final_states.append(final_state)

        x = self.final_norm(x)

        return x, final_states

    @torch.no_grad()
    def step(self, x: Tensor, states: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
        """Single step for autoregressive inference."""
        new_states = []

        for block, state in zip(self.blocks, states):
            x, new_state = block.step(x, state)
            new_states.append(new_state)

        x = self.final_norm(x)

        return x, new_states

    def init_states(self, batch_size: int, device: torch.device = None) -> list[Tensor]:
        """Initialize states for all blocks."""
        return [block.init_state(batch_size, device) for block in self.blocks]
