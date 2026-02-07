"""KSSM fused CUDA operations with autograd support.

Optimized CUDA kernels for H100 GPUs (SM 9.0).
The kssm._C extension must be compiled via `pip install -e .`.
"""

from torch import Tensor
from torch.autograd import Function

try:
    import kssm._C as _C
except ImportError as e:
    raise ImportError(
        "KSSM CUDA extension (kssm._C) not found. "
        "Build it with: pip install -e . "
        "(requires CUDA toolkit and an H100-compatible environment)"
    ) from e


class CUDAConv1dSiLUOp(Function):
    """PyTorch autograd Function for fused Conv1d + SiLU CUDA kernel."""

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        """Forward pass: depthwise conv1d + SiLU.

        Args:
            x: Input tensor (batch, seq, d_inner) in bfloat16
            weight: Conv weights (d_inner, 1, kernel_size) - kernel_size must be 4
            bias: Bias tensor (d_inner,)

        Returns:
            Output tensor (batch, seq, d_inner) in bfloat16
        """
        out = _C.conv1d_silu_fwd_cuda(x, weight, bias)
        ctx.save_for_backward(x, weight, bias)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Backward pass."""
        x, weight, bias = ctx.saved_tensors
        grad_x, grad_weight, grad_bias = _C.conv1d_silu_bwd_cuda(
            x, weight, bias, grad_out
        )
        return grad_x, grad_weight, grad_bias


def conv1d_silu_cuda(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Fused depthwise Conv1d + SiLU using CUDA kernel.

    Args:
        x: Input tensor (batch, seq, d_inner) in bfloat16
        weight: Conv weights (d_inner, 1, 4) - must be kernel_size=4
        bias: Bias tensor (d_inner,)

    Returns:
        Output tensor (batch, seq, d_inner) in bfloat16
    """
    return CUDAConv1dSiLUOp.apply(x, weight, bias)


__all__ = [
    "conv1d_silu_cuda",
]
