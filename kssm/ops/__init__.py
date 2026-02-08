"""KSSM fused CUDA operations with autograd support."""

from torch import Tensor
from torch.autograd import Function

import kssm._C as _C


class CUDAConv1dSiLUOp(Function):
    """PyTorch autograd Function for fused Conv1d + SiLU CUDA kernel."""

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        out = _C.conv1d_silu_fwd_cuda(x, weight, bias)
        ctx.save_for_backward(x, weight, bias)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x, weight, bias = ctx.saved_tensors
        grad_x, grad_weight, grad_bias = _C.conv1d_silu_bwd_cuda(
            x, weight, bias, grad_out
        )
        return grad_x, grad_weight, grad_bias


def conv1d_silu_cuda(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return CUDAConv1dSiLUOp.apply(x, weight, bias)


__all__ = ["conv1d_silu_cuda"]
