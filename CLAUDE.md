# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KSSM (Kinetic State Space Model) is a research project for a novel machine learning architecture targeting sequence modeling (LLMs, time series, audio). The architecture uses real-valued 2x2 block-diagonal state matrices to achieve rotary dynamics without complex numbers, combined with Cayley (bilinear) transform discretization for unconditional A-stability.

## Current State

**Status: Implementation complete, benchmarking in progress**

### Completed Implementation
- Core KSSM layer with Cayley discretization (`kssm/modules/kssm_layer.py`)
- Unified Mamba-style block architecture (`kssm/modules/kssm_block.py`)
- Fused Triton kernels for Cayley+Evolution scan (`kssm/kernels/`)
- Forward and backward pass with gradient checkpointing
- Language model wrapper (`kssm/model/language_model.py`)
- Nuclear initialization for long-range memory (`kssm/modules/init.py`)

### Benchmark Results
1. **Stability (Experiment 1)**: KSSM stable at LR=0.5, Mamba diverges at LR=0.5
2. **Passkey Retrieval (Experiment 2)**: 98% accuracy at seq_len=64, 97% at 128, needs larger model for 256+
3. **Memory Scaling (Experiment 3)**: Pending - `experiments/bench_vs_transformer.py`

### Parameter Efficiency
- KSSM achieves 0.98x Mamba parameters with unified block architecture
- Memory usage: 1.07x Mamba (within margin of error)
- Throughput gap: ~8x slower (Triton vs hand-optimized CUDA - expected)

## Architecture Key Concepts

**Core Innovation:** Pairs hidden states into 2x2 real-valued blocks that act as damped harmonic oscillators, enabling:
- Dynamic rotation (like RoPE but learnable/content-dependent)
- Dynamic damping (forgetting, like RWKV/Mamba)
- Purely real-valued operations (hardware efficient)

**Key Components:**
1. **State Structure:** Block-diagonal with 2x2 rotation-decay blocks parameterized by α (damping) and ω (frequency)
2. **Discretization:** Cayley transform with analytical O(1) inverse for each 2x2 block
3. **Parallelization:** Associative scan for O(log T) training

**Stability Guarantee:** Cayley transform ensures spectral radius ≤ 1 for any learnable parameters and any step size when α ≥ 0 (enforced via softplus).

## Key Files

### Core Implementation
- `kssm/modules/kssm_block.py` - Unified Mamba-style KSSM block (primary)
- `kssm/modules/kssm_layer.py` - Simple KSSM layer (legacy)
- `kssm/ops/scan_op.py` - Scan operation dispatcher (Triton/PyTorch)
- `kssm/kernels/evolution_fwd.py` - Fused Cayley+Evolution Triton kernel

### Models
- `kssm/model/backbone.py` - KSSMBackbone (stack of blocks)
- `kssm/model/language_model.py` - KSSMLMHeadModel (with embeddings + LM head)

### Experiments
- `experiments/bench_stability.py` - LR stability sweep vs Mamba
- `experiments/passkey.py` - Needle-in-haystack long-range retrieval
- `experiments/bench_vs_transformer.py` - Memory scaling vs FlashAttention
- `experiments/bench_vs_mamba.py` - Throughput comparison vs Mamba

### Examples
- `examples/induction_head.py` - Associative recall task (verified 100% accuracy)

## Running Experiments

```bash
# Stability test (LR sweep)
python experiments/bench_stability.py --task copy --d-model 128 --n-layers 2

# Passkey retrieval (needle in haystack)
python experiments/passkey.py --d-model 128 --n-layers 4 --seq-lengths 64 128 256 512

# Memory scaling vs Transformer
python experiments/bench_vs_transformer.py --batch-size 4 --seq-lengths 1024 2048 4096 8192

# Induction head (quick sanity check)
python examples/induction_head.py --seq-len 128 --n-steps 500
```

## Development Notes

### Nuclear Initialization
For long-range memory tasks, use nuclear_init:
```python
from kssm.modules.init import nuclear_init
for block in model.backbone.blocks:
    nuclear_init(block.mixer, long_memory=True)
```

### Triton Kernels
- Set `use_triton=True` in forward pass for fused kernels
- Fall back to `use_triton=False` for debugging or if Triton unavailable

### Testing
```bash
pytest tests/ -v
```
