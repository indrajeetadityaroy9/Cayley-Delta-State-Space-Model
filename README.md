# Cayley-Delta State Space Model (CD-SSM)

Long-context language modeling with stable dissipative state-space dynamics, gated delta-rule memory updates, and chunkwise SSD parallel scan.

Attempt to implement an attention-free causal language model that combines Cayley-discretized 2x2 dissipative dynamics per head, Delta-rule matrix memory with selective write/erase control, SSD-style chunkwise parallel scan for efficient long-sequence training.

The goal is to determine an integrated recurrent state-space design can match modern long-context requirements while retaining strong training/inference efficiency.

1. **Cayley-stable dynamics with controllable dissipation**  
   `cdssm/models/block.py` and `cdssm/ops/__init__.py` implement fused Cayley discretization (`cayley_vp_cuda`) with recurrence-gated eigenvalue modulation.
2. **Gated delta-rule memory in matrix state form**  
   `cdssm/models/ssd.py` implements a matrix memory state `(2 x D)` per head and chunkwise delta-rule updates.
3. **Adaptive timestep and layer-stratified spectral priors**  
   `cdssm/models/components.py` provides adaptive timestep (`adaptive_dt_cuda`) and `apply_spectral_init` covering timescales up to `context_length`.
4. **Utility-aware gating and auxiliary metabolic regularization**  
   `cdssm/models/block.py` and `cdssm/models/backbone.py` implement utility gating with EMA state energy and derived sparsity weight `1/log(vocab)^3`.
5. **CUDA-first kernelized implementation with autograd**  
   Custom kernels in `cdssm/csrc/kernels/` are exposed through `cdssm._C` and tested in `tests/` (correctness, gradients, performance, e2e stability).

## Unified Method: End-to-End Block Computation

Each `CDSSMBlock` (`cdssm/models/block.py`) executes one cohesive recurrent update pipeline:

1. Input projection into gate pathway, keys, and 2D values.
2. Fused depthwise conv + SiLU on gate pathway (`conv1d_silu_cuda`).
3. Dynamics/gate parameterization (`alpha`, `omega`, `dt`, recurrence gate, selection scalars, beta).
4. Fused Cayley discretization + variance-preserving scaling.
5. Chunkwise SSD scan with intra-chunk and inter-chunk CUDA kernels.
6. Query-based readout from matrix state, normalization, gated output projection, residual merge.

## References

1. **Transformers are SSMs / Mamba-2 (State Space Duality)**  
   Paper: https://arxiv.org/abs/2405.21060  
   Motivates SSD chunkwise training algorithm used in `cdssm/models/ssd.py`.
2. **Gated Delta Networks**  
   Paper: https://arxiv.org/abs/2410.04484  
   Informs gated delta-rule memory update design in `cdssm/models/block.py` and `cdssm/models/ssd.py`.
3. **Griffin (RG-LRU gating)**  
   Paper: https://arxiv.org/abs/2402.19427  
   Recurrence-gating ideas reflected in the learned recurrence gate in `cdssm/models/block.py`.
4. **S4 (Structured State Spaces)**  
   Paper: https://arxiv.org/abs/2111.00396  
   Foundational SSM motivation and long-range sequence modeling context.
5. **Symplectic Recurrent Neural Networks**  
   Paper: https://arxiv.org/abs/1909.13334  
   Inspires structure-preserving dynamics viewpoint (adapted here to dissipative Cayley dynamics).
6. **Attention when you need**  
   Paper: https://arxiv.org/abs/2407.07120  
   Relationship: motivates explicit utility/cost trade-off perspective used by auxiliary gating regularization.
