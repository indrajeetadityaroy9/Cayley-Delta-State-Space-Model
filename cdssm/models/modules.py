"""Neural network modules for CDSSM.

- Conv1dSiLU: Fused depthwise conv1d + SiLU using CUDA kernel
- RMSNorm: Root Mean Square Layer Normalization
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from cdssm.ops import CUDAConv1dSiLUOp


class Conv1dSiLU(nn.Module):
    """Depthwise Conv1d + SiLU using CUDA kernel."""

    def __init__(self, d_inner: int, kernel_size: int = 4):
        super().__init__()
        self.d_inner = d_inner
        self.kernel_size = kernel_size

        # Depthwise conv weights: (d_inner, 1, kernel_size)
        self.weight = nn.Parameter(torch.empty(d_inner, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(d_inner))
        self._init_weights()

    def _init_weights(self):
        """Initialize conv weights: Kaiming/He for depthwise with fan_in = kernel_size."""
        with torch.no_grad():
            nn.init.normal_(self.weight, std=1.0 / math.sqrt(self.kernel_size))
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        return CUDAConv1dSiLUOp.apply(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        eps: Normalization epsilon. Default derived from io_eps^2 (BF16 gradient
            representability constraint: rsqrt gradient O(eps^{-1/2}) < 1/io_eps).
    """

    def __init__(self, d_model: int, eps: float = 6.103515625e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).to(x.dtype) * self.weight
