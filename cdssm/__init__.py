"""CDSSM: Cayley-Delta State Space Model with A-stability via Cayley discretization.

Hardware target: NVIDIA H100 PCIe, 80GB HBM3, SM 9.0, 114 SMs.
H100 runtime optimizations are applied at import time.
"""

import os
import torch

# TF32 matmul: ~2x on fp32 ops (RMSNorm casts to fp32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# cuDNN benchmark: auto-tune conv algorithm selection
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Float32 matmul precision 'high': TF32 tensor cores for all matmul
torch.set_float32_matmul_precision("high")
# CUDA allocator: expandable_segments reduces fragmentation on 80GB HBM3
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)
