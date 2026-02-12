# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kinetic State Space Model (KSSM) — a pure SSM language model using Cayley-stable dissipative Hamiltonian dynamics with gated delta-rule selective memory. Physics-informed alternative to transformers with linear O(L) time complexity. Currently targets WikiText-103 with a GPT-2 tokenizer.

## Build & Run Commands

```bash
# Install (compiles CUDA extension kssm._C targeting sm_90/H100)
pip install -e .

# Train
python scripts/train.py --config configs/default.yaml

# Deterministic reproduction
./scripts/reproduce_results.sh
```

**Requirements**: PyTorch 2.4+ with CUDA 12.x, NVIDIA H100 GPU (sm_90 architecture).

No test suite or linter is configured in this repository.

## Architecture

### Data Flow

```
tokens → Embedding → KSSMBackbone (n KSSMBlocks) → RMSNorm → tied LM Head → logits
```

### Key Modules

- **`src/kssm/models/language_model.py`** — `KSSMLMHeadModel`: top-level wrapper with tied embedding weights
- **`src/kssm/models/backbone.py`** — `KSSMBackbone`: pre-norm residual stack of blocks, exposes `get_metabolic_loss()` for sparsity regularization
- **`src/kssm/models/kssm_block.py`** — `KSSMBlock`: core ~300-line block. Projects input into gates/keys/values, runs dynamics through SSD scan, applies dual gating (recurrence gate + utility gate), and produces output
- **`src/kssm/models/ssd.py`** — `SSDChunkwiseScan`: three-phase chunkwise parallel scan (intra-chunk sequential delta-rule, inter-chunk state propagation in FP32, broadcast correction). Uses `@torch.compile`
- **`src/kssm/models/components.py`** — Reusable building blocks: `RMSNorm`, `AdaptiveTimestep` (CFL-like dt), `Conv1dSiLU` (fused CUDA kernel wrapper), spectral initialization helpers

### CUDA Layer

- **`src/kssm/csrc/include/cayley_math.cuh`** — **CANONICAL** Cayley transform discretization. Single source of truth for converting continuous Hamiltonian `A=[[-α,ω],[-ω,-α]]` to discrete `A_bar` with guaranteed `|eigenvalue| ≤ 1`
- **`src/kssm/csrc/kernels/conv1d_silu.cu`** — Fused depthwise conv1d+SiLU kernel (forward + backward), bf16 I/O
- **`src/kssm/ops/__init__.py`** — Python autograd wrapper for the CUDA kernel

### Configuration

- **`src/kssm/config/defaults.py`** — `KSSMConfig` dataclass. Only 4 user-specified params (`d_model`, `n_layers`, `context_length`, `vocab_size`); all internal constants (head count, gating range, EMA decay, sparsity penalty, LR ratio) are derived from first principles
- **`configs/default.yaml`** — Training hyperparameters (lr=6e-4, batch=4, grad_accum=8, bf16, cosine schedule, 500-step warmup)

### Training

- **`src/kssm/training/trainer.py`** — `train_one_epoch()` with gradient accumulation and metabolic loss integration
- **`src/kssm/training/optim.py`** — Dual-rate param groups: SSM parameters get reduced LR via `1/sqrt(2*n_layers)` (T-Fixup scaling), plus cosine schedule
- **`src/kssm/__init__.py`** — Sets H100 runtime optimizations at import (TF32, cuDNN benchmark, CUDA allocator)

## Critical Design Invariants

- **Cayley discretization in `cayley_math.cuh` is the single source of truth.** Any Python-side Cayley logic must match it exactly. This guarantees A-stability (unconditional numerical stability) for any `α ≥ 0`.
- **Inter-chunk state accumulation in SSD must use FP32** to prevent precision loss across long sequences.
- **All "magic numbers" in KSSMConfig are derived, not tuned.** Changes to derived constants should preserve their mathematical justification (documented in comments).
- **Spectral initialization is layer-stratified**: early layers get short timescales (local patterns), deep layers get long timescales (document-level coherence), with 50% overlap between adjacent bands.
- **Variance-preserving initialization** uses T-Fixup scaling: `1/sqrt(2*n_layers)` for residual paths, `1/sqrt(d_model)` for embeddings, `1/sqrt(kernel_size)` for conv kernels.

## Known Limitations (from README)

1. **Readout bottleneck**: Each head's 2×D state matrix produces only a 2-vector readout (~3% expressivity vs attention). Future fix: increase Hamiltonian state dimension beyond 2.
2. **Sequential intra-chunk scan**: Delta-rule recurrence within 64-token chunks cannot be parallelized. Future fix: WY/UT decomposition or Triton kernels.
