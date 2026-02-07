"""KSSM: Kinetic State Space Model with A-stability via Cayley discretization.

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
# Deterministic off: seeds provide approximate reproducibility.
# For bit-exact reproduction (at ~10-15% throughput cost), set
# KSSM_STRICT_DETERMINISM=1 before importing kssm.
torch.backends.cudnn.deterministic = False
if os.environ.get("KSSM_STRICT_DETERMINISM", "0") == "1":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
# Float32 matmul precision 'high': TF32 tensor cores for all matmul
torch.set_float32_matmul_precision("high")
# CUDA allocator: expandable_segments reduces fragmentation on 80GB HBM3
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)
