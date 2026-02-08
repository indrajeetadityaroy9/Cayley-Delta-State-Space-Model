# Kinetic State Space Model (KSSM)

**Symplectic State Space Duality for Physics-Informed Long-Context Modeling**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)
[![H100 Optimized](https://img.shields.io/badge/Hardware-H100-green.svg)](https://www.nvidia.com/en-us/data-center/h100/)

**KSSM** is a research codebase implementing a **Pure State Space Model** that unifies Hamiltonian dynamics with efficient parallel scanning. By embedding symplectic structures directly into the state-space recurrence, KSSM achieves lossless long-term memory within a fully parallelizable framework, eliminating the need for attention mechanisms or hybrid architectures.

> **Hardware Note:** This implementation is highly optimized for **NVIDIA H100 (Sm90)** GPUs, utilizing `torch.compile` and tensor core instructions for the State Space Duality (SSD) algorithm.

## Research Objectives

Standard State Space Models (SSMs) like Mamba have revolutionized efficient sequence modeling but often rely on heuristic diagonal approximations. KSSM introduces a rigorous physics-informed prior to solve the stability-expressivity tradeoff:

1.  **Symplectic State Space Duality (S3D):** We generalize the Mamba-2 SSD algorithm to support **Hamiltonian Systems**. By restricting the state matrix $A$ to the Lie algebra of the symplectic group, we guarantee energy conservation (determinant = 1) in the limit of zero damping. This solves the "vanishing memory" problem without requiring $O(N^2)$ attention.
2.  **Kinetic A-Stability:** We utilize the **Cayley Transform** for discretization. Unlike Zero-Order Hold (ZOH), this maps the left half-plane to the unit disk unconditionally, ensuring stability even for stiff, high-frequency oscillatory dynamics.
3.  **Parameter-Free Scale Invariance:** An **Adaptive Timestep** mechanism ($dt \propto 1/|\omega|$) automatically normalizes dynamics to the system's natural frequency, removing the need for manual timescale initialization.

## Architecture

KSSM is a **Homogeneous Pure SSM** architecture. It relies exclusively on the **KSSM Block** for both mixing and memory.

### The KSSM Block
*   **Dynamics:** 2x2 Block-Diagonal Hamiltonian System ($q, p$ pairs) evolving via $\dot{h} = (J - R)h$.
*   **Algorithm:** **Chunkwise Parallel Scan (SSD)**.
    *   **Intra-Chunk:** Parallel associative scan via Tensor Cores.
    *   **Inter-Chunk:** Sequential passing of symplectic states.
    *   **Complexity:** $O(L)$ compute, $O(L/Q)$ sequential steps.
*   **Gating:**
    *   **Zero-Damping Gate (ZDG):** Dynamically modulates damping $\alpha \to 0$ for lossless storage.
    *   **Variance-Preserving Input:** Griffin-style gating ensures stable signal propagation at infinite depth.

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/kinetic-state-space-model.git
cd kinetic-state-space-model

# Install dependencies (requires CUDA 12.x)
pip install -e .
```

## Usage

### Configuration
The model is defined by `KSSMConfig`. The architecture is purely defined by `d_model` and `n_layers`.

```python
from kssm.config import KSSMConfig
from kssm.model.backbone import KSSMBackbone

# Initialize a purely kinetic model
config = KSSMConfig(
    d_model=768,
    n_layers=24
)

model = KSSMBackbone(config)
```

### Reproduction
Canonical execution path (WikiText-103):

```bash
python reproduce.py --epochs 20 --context-length 1024
```

## References

This project builds upon and integrates ideas from the following foundational papers:

*   **Mamba-2 / State Space Duality:** Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
*   **Symplectic Recurrent Neural Networks:** Chen, Z., et al. (2020). *Symplectic Recurrent Neural Networks*. [arXiv:1909.13334](https://arxiv.org/abs/1909.13334)
*   **Structured State Spaces (S4):** Gu, A., et al. (2022). *Efficiently Modeling Long Sequences with Structured State Spaces*. [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)

## Paper Basis For Recent Optimizations

Recent implementation changes were informed by the following arXiv papers:

* **Mamba:** Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
* **Mamba-2 / SSD:** Dao, T., & Gu, A. (2024). *Transformers are SSMs*. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
* **PackMamba:** (2024). *PackMamba: High-throughput Mamba for variable-length sequences*. [arXiv:2408.03865](https://arxiv.org/abs/2408.03865)
* **Falcon-Mamba-7B:** (2024). *The First Competitive Attention-free 7B Language Model*. [arXiv:2410.05355](https://arxiv.org/abs/2410.05355)
* **HGRN2:** Qin, Z., et al. (2024). *Gated Linear RNNs with State Expansion*. [arXiv:2404.07904](https://arxiv.org/abs/2404.07904)
* **RULER:** Hsieh, C.-Y., et al. (2024). *What’s the Real Context Size of Your Long-Context Language Models?* [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)

## Important Constraint

Current optimizations improve performance and stability in the implemented path, but they do **not** by themselves prove that all performance issues are resolved. Final claims require workload-specific profiling on the target hardware (H100/SM90) under real training and inference settings.

## License

MIT License. See `LICENSE` for details.
