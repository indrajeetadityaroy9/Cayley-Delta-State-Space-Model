# Kinetic State Space Model (KSSM)

**Cayley-Stable Dissipative Hamiltonian Dynamics with Delta-Rule Selective Memory for Long-Context Language Modeling**

## Abstract

State space models (SSMs) achieve linear-time sequence modeling but face a fundamental tension: diagonal state matrices scale efficiently yet lack the representational capacity for precise memory management, while dense matrices enable selective state updates but resist parallelization. KSSM resolves this by embedding **2x2 dissipative Hamiltonian blocks** into a chunkwise parallel scan. The continuous dynamics $A = [[-\alpha, \omega], [-\omega, -\alpha]]$ are discretized via the **Cayley transform**, which guarantees that eigenvalues of the discrete transition matrix lie strictly within the unit disk for any $\alpha > 0$ (A-stability), without constraining the learning rate or requiring post-hoc clipping. Within this stable recurrence, a **gated delta rule** enables selective erasure of individual state associations, a **learned recurrence gate** provides input-dependent retention control independent of the Hamiltonian dynamics, and **layer-stratified spectral priors** assign depth-dependent frequency bands so that early layers capture local syntax while deep layers maintain document-level coherence. The result is a pure SSM — no attention layers, no hybrid MLP blocks — that combines physics-guaranteed stability with precise, learnable memory management in $O(L)$ time and $O(L/Q)$ sequential steps.

## 1. Research Objectives

Standard SSMs (S4, Mamba, Mamba-2) rely on diagonal or near-diagonal state matrices with heuristic initialization. This creates three problems:

1. **Stability is fragile.** Diagonal eigenvalues near the unit circle require careful initialization and gradient clipping. Small perturbations during training can push eigenvalues outside the disk, causing state explosion.
2. **Memory updates are coarse.** Hebbian outer-product injection ($h \leftarrow Ah + vk^\top$) can only accumulate — it cannot selectively erase a specific association without uniformly decaying the entire state.
3. **Timescale allocation is uniform.** All layers receive identical spectral priors, forcing the model to learn timescale specialization from scratch.

KSSM addresses each through a single integrated architecture.

## 2. Method

### 2.1 Cayley-Stable Dissipative Hamiltonian Dynamics

The continuous state matrix for each head is a 2x2 dissipative Hamiltonian:

$$A = \begin{bmatrix} -\alpha & \omega \\ -\omega & -\alpha \end{bmatrix}, \quad \alpha \geq 0$$

where $\alpha$ controls damping (energy dissipation) and $\omega$ controls rotation (oscillatory memory). Discretization uses the **Cayley transform**:

$$\bar{A} = (I - \tfrac{\Delta t}{2} A)^{-1}(I + \tfrac{\Delta t}{2} A)$$

This yields eigenvalues with magnitude $|\lambda|^2 = \frac{(1 - \tau\alpha)^2 + (\tau\omega)^2}{(1 + \tau\alpha)^2 + (\tau\omega)^2} \leq 1$ unconditionally for $\alpha \geq 0$ (where $\tau = \Delta t / 2$), providing A-stability without any post-hoc correction. At $\alpha = 0$, the dynamics are exactly volume-preserving ($|\det(\bar{A})| = 1$), recovering energy-conserving rotation.

### 2.2 Gated Delta-Rule State Update

Following Gated DeltaNet (Yang et al., 2024), the recurrence replaces Hebbian accumulation with selective erasure:

$$h_t = \bar{A}_t \left( h_{t-1} - \beta_t (k_t^\top h_{t-1}) k_t \right) + \beta_t \, v_t k_t^\top \Delta t_t$$

where $\beta_t = \sigma(\text{beta\_proj}(x_t)) \in [0,1]$ is a learned gate. The term $\beta_t (k_t^\top h_{t-1}) k_t$ erases the projection of the current state onto the key direction before injecting the new association $v_t k_t^\top$. This gives the model precise control over which memories to overwrite, operating within the Cayley-stable evolution.

### 2.3 Learned Recurrence Gate (Griffin RG-LRU)

A learned gate $r_t = \sigma(\text{recurrence\_gate}(x_t))$ modulates the effective eigenvalue decay independent of the Hamiltonian parameters (De et al., 2024):

$$|\lambda_{\text{eff}}|^2 = |\lambda(\bar{A})|^{2 \cdot c \cdot r_t}, \quad c = \ln(T)$$

where $T$ is the context length. The gating range $c$ is derived from the sequence length rather than set manually — longer contexts require stronger flushing capability. For $T = 8192$, $c \approx 9.0$; for $T = 1024$, $c \approx 6.9$. When $r_t \to 0$, the effective eigenvalue approaches 1 (state held). When $r_t \to 1$, the decay is amplified by $c$ (state flushed). A-stability is preserved for all $c > 0$ since $|\lambda| \leq 1$ implies $|\lambda|^{cr} \leq 1$. The variance-preserving input scale $\sqrt{1 - |\lambda_{\text{eff}}|^2}$ ensures signal magnitude is preserved across the recurrence regardless of the gate setting.

### 2.4 State-Conditioned Utility Gating

A utility gate $u_t \in [0,1]$ suppresses input injection when the current input is irrelevant to the memory state, enabling "metabolic coasting" (Boominathan et al., 2025):

$$u_t = \sigma\!\left(\text{utility\_gate}(x_t) + \text{state\_gate\_proj}(\bar{E})\right)$$

where $\bar{E}$ is a per-head exponential moving average of state energy, updated each forward pass during training:

$$\bar{E}_h \leftarrow \gamma_h \, \bar{E}_h + (1 - \gamma_h) \, \|h\|^2, \quad \gamma_h = \text{clamp}\!\left(1 - \tfrac{1}{\tau_h}, \; 0.9, \; 0.999\right)$$

The EMA decay $\gamma_h$ is derived from each head's spectral initialization timescale $\tau_h$ — fast-decaying heads (short $\tau$) update their energy estimate quickly, while long-memory heads track energy over many steps. This provides a stable, per-head signal of current memory utilization without cross-batch leakage and without a manually tuned decay constant.

The gate modulates $V$ before injection: when $u_t \approx 0$, the existing state is preserved without update. An L1 penalty $\lambda = 1 / \ln(V)^3$ (where $V$ is the vocabulary size) on $\mathbb{E}[u_t]$ encourages sparsity. The penalty is derived from the expected initial cross-entropy scale, providing gentle, scale-invariant pressure without manual tuning.

### 2.5 Layer-Stratified Spectral Initialization

Each layer $\ell \in \{0, \ldots, L-1\}$ receives a depth-dependent band of the log-timescale range $[\log(1), \log(T)]$ where $T$ is the context length:

$$\text{band}_\ell = \left[\log(1) + f_\ell \cdot \tfrac{R}{2}, \; \log(1) + f_\ell \cdot \tfrac{R}{2} + \tfrac{R}{2}\right], \quad f_\ell = \frac{\ell}{L-1}, \; R = \log(T)$$

Each band covers 50% of the total range and slides with depth. Within a band, $n_{\text{heads}}$ timescales are log-spaced and mapped to initial $\alpha = 1/\tau$ (damping) and $\omega$ (frequency) values via inverse-softplus bias initialization. Early layers ($\ell = 0$) receive short timescales (high $\alpha$, fast decay) for local pattern matching; deep layers ($\ell = L-1$) receive long timescales (low $\alpha$, slow decay) for document-level coherence.

### 2.6 Adaptive Timestep

The timestep $\Delta t$ adapts to the characteristic frequency of each head:

$$\Delta t = \frac{\text{softplus}(c_h)}{\alpha + |\omega| + \epsilon} + \text{softplus}(\text{selection\_dt}(x_t))$$

where $c_h$ is a learnable per-head scale. This CFL-like condition ensures that fast-oscillating heads take small steps (preserving phase accuracy) while slowly-varying heads take large steps (avoiding unnecessary computation per unit of modeled time). A smooth safety cap prevents precision collapse when $\omega \to 0$ in bfloat16.

### 2.7 Position Encoding via Omega Modulation

Position information is injected directly into the oscillatory dynamics rather than through additive embeddings:

$$\omega'_t = \omega_t + t \cdot f_h, \quad f_h = \frac{1}{10000^{h/H}}$$

where $f_h$ are RoPE-like per-head frequencies. This modulates the rotation rate as a function of absolute position, providing implicit positional encoding within the state evolution.

### 2.8 Chunkwise Parallel Scan (SSD)

The sequence is split into chunks of size $Q = 64$. Within each chunk, the delta-rule recurrence runs sequentially ($O(Q)$ steps). Across chunks, cumulative Cayley transition products propagate state via sequential recurrence ($O(L/Q)$ steps). A broadcast correction combines local and global states:

$$h_t^{\text{true}} = h_t^{\text{local}} + \left(\prod_{s=0}^{t} \bar{A}_s\right) h_{k}^{\text{global}}$$

Inter-chunk correction uses base $\bar{A}$ products (without delta erasure) — an acceptable approximation since most associations form and are erased within a single chunk of 64 steps. Total complexity is $O(L)$ time with $O(L/Q)$ sequential steps. Both the intra-chunk and inter-chunk scans use FP32 state accumulation with `@torch.compile` for JIT optimization.

### 2.9 Parameter-Free Internal Constants

All internal constants are derived from the minimal configuration surface — no magic numbers:

| Internal Constant | Derivation | Replaces |
|-------------------|------------|----------|
| Gating range $c$ | $\ln(\text{context\_length})$ | Griffin's fixed $c = 8$ |
| EMA decay $\gamma_h$ | $\text{clamp}(1 - 1/\tau_h, \; 0.9, \; 0.999)$ per head | Uniform $0.99$ |
| Sparsity penalty $\lambda$ | $1 / \ln(\text{vocab\_size})^3$ | Fixed $10^{-3}$ |
| SSM learning rate ratio | $1 / \sqrt{2 \cdot n_{\text{layers}}}$ | Fixed $0.1\times$ |
| PPL display clamp | $\ln(\text{vocab\_size})$ | Fixed $20.0$ |

Each derivation is grounded in a clear principle: gating range scales with sequence length (longer contexts need stronger flushing), EMA decay matches each head's characteristic timescale from spectral initialization, the sparsity penalty normalizes against the expected initial cross-entropy, and the SSM learning rate ratio follows T-Fixup depth scaling to preserve spectral initialization structure.

## 3. Architecture

KSSM is a **homogeneous pure SSM**. The backbone is a stack of identical KSSM Blocks with pre-norm residual connections. There are no attention layers, MLP blocks, or hybrid components.

### KSSM Block

Each block applies the following operations in sequence:

```
Input x: (B, L, d_model)
    |
    v
[RMSNorm] ──> [in_proj] ──> split into z (gate), x_gate, K (D-dim), V (2D)
                                              |
                                              v
                                   [Conv1dSiLU on x_gate] ──> x_conv
                                              |
                                              v
                 ┌────────────────────────────┼────────────────┐
                 |                            |                 |
         [dynamics_proj]              [selection_B/C/dt]  [utility_gate]
          alpha, omega                  K *= sel_B         u_t (sparse)
                 |                      Q *= sel_C              |
                 v                      dt += sel_dt            |
         [AdaptiveTimestep]                   |                 v
              dt                              |        [state_gate_proj]
                 |                            |         (EMA energy feedback)
                 v                            |                 |
         [recurrence_gate]                    |                 v
              r_t                             |   V_gated = V * vp_scale * u_t
                 |                            |                 |
                 v                            v                 v
         [VP Scale Gated]              [beta_proj]     ┌───────┘
              vp_scale                    beta_t       |
                 |                            |        |
                 v                            v        v
              ╔═══════════════════════════════════════════╗
              ║    SSD Chunkwise Parallel Scan            ║
              ║    K: (B,L,H,D), V: (B,L,H,2)           ║
              ║    State h: (H, 2, D) matrix memory      ║
              ║    Cayley + Delta Rule + r_gate on A_bar  ║
              ║    Returns: Y (B,L,H,2,D), chunk_states  ║
              ╚═══════════════════════════════════════════╝
                          |
                          v
   [L2-norm Q] ──> readout: einsum(Y, Q) ──> (B,L,H,2) ──> [readout_proj]
                          |
                          v
              [GroupNorm] ──> [z-gate (SiLU)] ──> [+ D skip (x_conv)]
                          |
                          v
                    [out_proj]
                          |
                          v
                  residual + output
```

**Decoupled gate pathway:** The `x_gate` branch feeds the conv and all dynamics/selection/gate projections, while `K` (D-dimensional keys) and `V` (2D Hamiltonian values) are projected directly from `in_proj`. Keys and queries are L2-normalized for delta-rule training stability (per Gated DeltaNet ablations).

### Configuration

Architecture dimensions and all internal constants are derived from four integers with no manual tuning:

| Parameter | Derivation | Default |
|-----------|-----------|---------|
| `d_model` | User-specified | 768 |
| `n_layers` | User-specified | 12 |
| `context_length` | Upper bound of spectral prior | 8192 |
| `vocab_size` | Set from tokenizer | 50257 |
| `d_inner` | `2 * d_model` | 1536 |
| `head_dim` | Fixed at 64 (H100 Tensor Core alignment) | 64 |
| `n_heads` | `d_inner / head_dim` | 24 |

### Parameter Count

For the default configuration (768/12):
- Per block: `in_proj` + `dynamics_proj` + `conv1d` + `selection_{B,C,dt}` + `beta_proj` + `recurrence_gate` + `utility_gate` + `state_gate_proj` + `Q_proj` + `out_proj` + norms
- Total parameters scale as $\approx 12 d_{\text{model}}^2 \times n_{\text{layers}}$

## 4. Repository Structure

```
kinetic-state-space-model/
├── configs/
│   └── default.yaml              # Training hyperparameters
├── scripts/
│   ├── train.py                  # Training entry point
│   └── reproduce_results.sh      # Deterministic reproduction
└── src/kssm/
    ├── config/
    │   └── defaults.py           # KSSMConfig dataclass, derived constant functions
    ├── data/
    │   └── datasets.py           # WikiText-103 with GPT-2 tokenizer
    ├── models/
    │   ├── backbone.py           # KSSMBackbone: block stack + metabolic loss
    │   ├── kssm_block.py         # KSSMBlock: full block implementation
    │   ├── language_model.py     # KSSMLMHeadModel: embedding + backbone + tied LM head
    │   ├── ssd.py                # SSD chunkwise scan with delta-rule recurrence
    │   └── components.py         # AdaptiveTimestep, Conv1dSiLU, VP scaling, spectral init
    ├── ops/
    │   └── __init__.py           # Fused CUDA Conv1d+SiLU with autograd
    ├── csrc/
    │   ├── binding.cpp           # Pybind11 entry point
    │   ├── kernels/
    │   │   └── conv1d_silu.cu    # Fused depthwise conv1d + SiLU kernel (Sm90)
    │   └── include/
    │       ├── cayley_math.cuh   # Cayley transform math utilities
    │       ├── common.cuh        # CUDA common definitions
    │       └── reduction.cuh     # Warp/block reduction primitives
    ├── training/
    │   ├── trainer.py            # Training loop with metabolic loss integration
    │   └── optim.py              # Dual-rate param groups + cosine schedule
    ├── evaluation/
    │   └── evaluator.py          # Validation loss computation
    └── utils/
        ├── seeding.py            # Deterministic seeding across all RNGs
        └── checkpointing.py      # Checkpoint save with full reproducibility metadata
```

## 5. Installation

Requires **PyTorch 2.4+** with **CUDA 12.x** (Sm90 for H100 tensor core kernels).

```bash
git clone https://github.com/your-org/kinetic-state-space-model.git
cd kinetic-state-space-model
pip install -e .
```

The `pip install -e .` step compiles the fused CUDA extension (`kssm._C`) for the Conv1d+SiLU kernel. Compilation targets `sm_90` (H100) by default; modify `setup.py` for other architectures.

## 6. Usage

### Quick Start

```python
from kssm.config.defaults import KSSMConfig
from kssm.models.backbone import KSSMBackbone

config = KSSMConfig(d_model=768, n_layers=12, context_length=8192)
model = KSSMBackbone(config)  # Layer-stratified spectral priors applied automatically
```

### Training

```bash
python scripts/train.py --config configs/default.yaml
```

Default configuration (`configs/default.yaml`):

| Hyperparameter | Value |
|---------------|-------|
| Dataset | WikiText-103 (GPT-2 tokenizer, vocab 50257) |
| Context length | 1024 |
| Batch size | 4 (x8 gradient accumulation = 32 effective) |
| Optimizer | AdamW ($\beta_1=0.9$, $\beta_2=0.95$, wd=0.1, fused) |
| Base LR | $6 \times 10^{-4}$ (SSM params at $1/\sqrt{2 \cdot n_{\text{layers}}} \approx 0.2\times$) |
| Schedule | Cosine decay with 500-step linear warmup |
| Precision | bfloat16 |
| Epochs | 20 |

The SSM parameter group (dynamics projections, gates, selection layers) receives a reduced learning rate derived from model depth: $\text{lr}_{\text{SSM}} = \text{lr}_{\text{base}} / \sqrt{2 \cdot n_{\text{layers}}}$ (T-Fixup aligned). This preserves spectral initialization structure during early training and scales naturally with depth.

### Reproduction

```bash
./scripts/reproduce_results.sh
```

Deterministic seeding covers Python, NumPy, PyTorch, CUDA, and data loader workers.

## 7. Core Contributions

1. **Cayley-Stable State Space Duality.** The first chunkwise parallel scan adapted for 2x2 dissipative Hamiltonian systems. The Cayley transform provides unconditional A-stability — eigenvalues within the unit disk for any $\alpha \geq 0$ — without requiring diagonal approximation, gradient clipping, or eigenvalue projection.

2. **Delta-Rule Selective Erasure within Cayley Dynamics.** Integration of gated delta-rule state updates (selective association erasure) into the Cayley-stable recurrence. This combines the stability guarantee of physics-informed dynamics with the memory precision of DeltaNet, operating within the chunkwise parallel scan framework.

3. **Learned Dual Gating.** A Griffin RG-LRU-style recurrence gate provides input-dependent retention control orthogonal to the Hamiltonian decay, complemented by a state-conditioned utility gate that suppresses irrelevant updates based on current memory contents.

4. **Layer-Stratified Spectral Priors.** Depth-dependent frequency band initialization assigns short timescales to early layers and long timescales to deep layers, with 50% overlap between adjacent bands. Combined with adaptive timestep normalization, this provides scale-invariant dynamics across the full timescale range.

## 8. Known Limitations

### Readout Capacity Bottleneck

Each head maintains a 2×D matrix memory, but readout retrieves a **2-vector per head** via $\text{einsum}(Y, Q) \in \mathbb{R}^{H \times 2}$. With the default configuration ($H = 24$), this produces a 48-dimensional vector that `readout_proj` expands to $d_{\text{inner}} = 1536$. The maximum rank of this projection is 48 — roughly **3% of the per-token expressivity** of standard multi-head attention, which produces $H \times d_{\text{head}} = 24 \times 64 = 1536$ values directly.

The model compensates through temporal integration: the matrix memory accumulates information over many timesteps, so the effective capacity is much larger than a single readout step suggests. However, for tasks requiring high single-step information throughput (e.g., large lookup tables), this bottleneck may be limiting.

**Potential mitigation:** Increase the Hamiltonian state dimension from 2 to a configurable `d_state` (e.g., 8 or 16). This would change $\bar{A}$ from $2 \times 2$ to $d_{\text{state}} \times d_{\text{state}}$, requiring block-diagonal or structured parameterization to maintain efficient Cayley discretization. The readout would then produce $H \times d_{\text{state}}$ values per token, directly reducing the bottleneck.

### Sequential Intra-Chunk Scan

The intra-chunk delta-rule scan (`_intra_chunk_scan_delta`) uses a sequential Python loop over $C = 64$ timesteps per chunk. While `@torch.compile` enables JIT fusion and eliminates Python overhead, it cannot parallelize the inherent sequential dependency of the delta-rule recurrence:

$$h_t = \bar{A}_t \left( h_{t-1} - \beta_t (k_t^\top h_{t-1}) k_t \right) + \beta_t \, v_t k_t^\top \Delta t_t$$

Each step depends on $h_{t-1}$, preventing within-chunk parallelism. SOTA implementations of linear attention with delta rule (Gated DeltaNet) use the **WY/UT decomposition** to reformulate the sequential scan as a series of matrix multiplications amenable to GPU tensor cores, or employ custom **Triton kernels** for hardware-parallel execution.

This is a **throughput limitation**, not a correctness issue. The inter-chunk recurrence ($O(L/Q)$ sequential steps) and broadcast correction are already efficient. Replacing the intra-chunk loop with a WY-decomposition or Triton kernel would be the primary optimization for training speed at scale.

## 9. References

| Paper | Relationship to KSSM |
|-------|---------------------|
| Dao & Gu (2024). [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060). | **SSD chunkwise parallel scan.** KSSM adapts the Mamba-2 chunk-level scan for 2x2 block matrices with Cayley discretization. The intra-chunk/inter-chunk/correction decomposition is preserved; the scan kernel is replaced with a sequential delta-rule recurrence. |
| Yang et al. (2024). [Gated Delta Networks: Improving Mamba-2 with Delta Rule](https://arxiv.org/abs/2412.06464). | **Delta-rule state update.** KSSM integrates the gated delta rule ($h \leftarrow A(h - \beta k k^\top h) + \beta v k^\top$) into the Cayley recurrence, enabling selective association erasure within A-stable dynamics. |
| De et al. (2024). [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427). | **Learned recurrence gate + variance-preserving scale.** The RG-LRU mechanism ($|\lambda_{\text{eff}}| = |\lambda|^{c \cdot r_t}$) provides learned retention control. KSSM derives both the base eigenvalue from Cayley discretization and the gating range $c = \ln(T)$ from context length, replacing Griffin's fixed $c = 8$. |
| Boominathan et al. (2025). [Attention When You Need](https://arxiv.org/abs/2501.07440). | **Utility gating.** The metabolic gate concept — suppressing state updates when input is irrelevant — motivates KSSM's state-conditioned utility gate with L1 sparsity regularization. |
| Gu et al. (2022). [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396). | **Structured state spaces foundation.** S4 established the HiPPO initialization and parallel scan framework. KSSM replaces diagonal/NPLR structure with 2x2 Hamiltonian blocks and replaces HiPPO with layer-stratified spectral priors. |
| Chen et al. (2020). [Symplectic Recurrent Neural Networks](https://arxiv.org/abs/1909.13334). | **Hamiltonian structure in recurrence.** SymRNN demonstrated that Hamiltonian priors improve long-term gradient flow. KSSM extends this with dissipation ($\alpha > 0$), Cayley discretization (replacing leapfrog), and integration into the SSD parallel scan. |
