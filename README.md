# Kinetic State Space Model (KSSM)

**Symplectic State Space Duality for Physics-Informed Long-Context Modeling**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)
[![H100 Optimized](https://img.shields.io/badge/Hardware-H100-green.svg)](https://www.nvidia.com/en-us/data-center/h100/)

**KSSM** is a research codebase implementing a **Pure State Space Model** that unifies Hamiltonian dynamics with efficient parallel scanning. By embedding symplectic structures directly into the state-space recurrence, KSSM achieves lossless long-term memory within a fully parallelizable framework, eliminating the need for attention mechanisms or hybrid architectures.

> **Hardware Note:** This implementation is highly optimized for **NVIDIA H100 (Sm90)** GPUs, utilizing `torch.compile` and tensor core instructions for the State Space Duality (SSD) algorithm.

## 1. Project Overview & Research Objectives

Standard State Space Models (SSMs) like Mamba have revolutionized efficient sequence modeling but often rely on heuristic diagonal approximations. KSSM introduces a rigorous physics-informed prior to solve the **stability-expressivity tradeoff**:

*   **Problem:** Conventional RNNs and SSMs struggle to maintain stable signal propagation over ultra-long contexts due to the "vanishing gradient/memory" problem, or they require $O(N^2)$ attention mechanisms that scale poorly.
*   **Solution:** KSSM imposes **Symplectic Structure** on the state transition. In a Hamiltonian system, phase space volume is conserved (Liouville's Theorem). By discretizing this structure using the Cayley Transform, we guarantee that the state evolution remains on the unit circle (energy conserving) or spirals strictly inward (stable), never exploding.

### Core Contributions
1.  **Symplectic State Space Duality (S3D):** We generalize the Mamba-2 SSD algorithm to support **2x2 Hamiltonian Systems**. By restricting the state matrix $A$ to the Lie algebra of the symplectic group, we guarantee energy conservation (determinant = 1) in the limit of zero damping.
2.  **Kinetic A-Stability:** We utilize the **Cayley Transform** for discretization. Unlike Zero-Order Hold (ZOH), this maps the left half-plane to the unit disk *unconditionally*, ensuring stability even for stiff, high-frequency oscillatory dynamics.
3.  **Parameter-Free Scale Invariance:** An **Adaptive Timestep** mechanism ($dt \propto 1/|\omega|$) automatically normalizes dynamics to the system's natural frequency, removing the need for manual timescale initialization (`dt_min`, `dt_max`, `dt_init`).
4.  **Universal Spectral Priors:** Instead of learned initialization, we initialize the model with a fixed, logarithmic distribution of timescales spanning $[1, L_{context}]$, ensuring all frequencies are captured at initialization.

## 2. System Architecture

KSSM is a **Homogeneous Pure SSM** architecture. It relies exclusively on the **KSSM Block** for both mixing and memory, without hybrid attention or MLP layers.

### The KSSM Block
The core unit `src/kssm/models/kssm_block.py` composes the following operations:

1.  **Adaptive Timestep:** Computes $\Delta t = \text{softplus}(c) / (\alpha + |\omega| + \epsilon)$.
2.  **Variance-Preserving Gating:** Modulates input injection $B$ using Griffin-style RG-LRU logic to ensure the signal variance remains constant regardless of sequence depth.
3.  **Chunkwise Parallel Scan (SSD):**
    *   **Intra-Chunk:** Fused dual-form recurrence computed via Tensor Core matmuls.
    *   **Inter-Chunk:** Sequential recurrence of the hidden state $h_t$ across chunks.
    *   **Complexity:** $O(L)$ time, $O(L/Q)$ sequential steps.
4.  **Utility Gating:** A sparse "metabolic" gate $u_t$ that can suppress state updates (coasting) when the input is irrelevant, regularized by `METABOLIC_LAMBDA`.

### Directory Structure
```
src/kssm/
├── config/       # Configuration schemas (parameter-free)
├── data/         # Data loading and tokenization
├── models/       # Core architecture (Backbone, KSSMBlock, SSD)
├── ops/          # Fused CUDA kernels
├── training/     # Training loop and optimization
└── utils/        # Seeding and checkpointing
```

## 3. Installation

This codebase requires an environment with **PyTorch 2.1+** and **CUDA 12.x** (for H100 features).

```bash
# Clone repository
git clone https://github.com/your-org/kinetic-state-space-model.git
cd kinetic-state-space-model

# Install dependencies and compile CUDA extensions
pip install -e .
```

## 4. Usage

### Quick Start
Initialize a model with zero manual parameter tuning. The configuration is derived purely from `d_model` and `n_layers`.

```python
from kssm.config.defaults import KSSMConfig
from kssm.models.backbone import KSSMBackbone

# Initialize a purely kinetic model
config = KSSMConfig(
    d_model=768,
    n_layers=24,
    context_length=8192
)

# Automatically uses Universal Spectral Priors
model = KSSMBackbone(config)
```

### Reproduction
To reproduce the main WikiText-103 results reported in the paper (or internal benchmarks), use the provided reproduction script. This runs a deterministic training path ensuring valid comparisons.

```bash
./scripts/reproduce_results.sh
```

Key hyperparameters are defined in `configs/default.yaml`.

## 5. References

This project builds upon and integrates ideas from the following foundational papers:

*   **Mamba-2 / State Space Duality:** Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
*   **Symplectic Recurrent Neural Networks:** Chen, Z., et al. (2020). *Symplectic Recurrent Neural Networks*. [arXiv:1909.13334](https://arxiv.org/abs/1909.13334)
*   **Structured State Spaces (S4):** Gu, A., et al. (2022). *Efficiently Modeling Long Sequences with Structured State Spaces*. [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)
*   **Griffin (RG-LRU):** De, S., et al. (2024). *Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models*. [arXiv:2402.19427](https://arxiv.org/abs/2402.19427)
*   **Attention when you need:** Boominathan, L., et al. (2025). *Attention when you need*. [arXiv:2501.07440](https://arxiv.org/abs/2501.07440)

## License

MIT License. See `LICENSE` for details.