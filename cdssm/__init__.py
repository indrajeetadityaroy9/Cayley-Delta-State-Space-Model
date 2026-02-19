"""CDSSM: Cayley-Delta State Space Model with A-stability via Cayley discretization."""

import os

import torch

# H100 runtime optimizations
# Hardware target: NVIDIA H100 PCIe, 80GB HBM3, SM 9.0, 114 SMs.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("high")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
