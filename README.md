# Cayley-Delta State Space Model (CD-SSM)

An attention-free recurrent language model combining Cayley-discretized dissipative dynamics, delta-rule matrix memory, and chunkwise parallel scan. CD-SSM targets long-context causal language modeling with unconditional numerical stability (A-stability) and linear-time training via State Space Duality

## Method

CD-SSM integrates three mechanisms into a single recurrent block:

**Cayley-stable dissipative dynamics.** Each head maintains a 2D state vector evolving under a continuous rotation-damping system `dz/dt = Az` where `A = [[-alpha, omega], [-omega, -alpha]]` with `alpha >= 0`. The Cayley transform `A_bar = (I - tau*A)^{-1}(I + tau*A)` discretizes this system with a provable guarantee: eigenvalues of `A_bar` lie inside the unit disk for any `alpha >= 0` and any step size `dt`, ensuring unconditional stability across arbitrary sequence lengths. A learned recurrence gate modulates the effective eigenvalue magnitude per head per timestep, following the RG-LRU mechanism.

**Delta-rule matrix memory.** Each head stores a `(2 x D)`-dimensional matrix state, updated via a gated selective erase-write rule: `h[t] = A_bar[t] (h[t-1] - beta * (h @ k) k^T) + beta * v k^T`. The erase term `(h @ k) k^T` selectively removes content along the current key direction before writing new associations. Write strength `beta = sigmoid(beta_raw) * sigmoid(sel_B)` and read strength `sel_C = sigmoid(sel_C_raw)` are input-dependent. Keys and queries are L2-normalized for delta-rule stability.

**Chunkwise parallel scan (SSD).** Training uses the State Space Duality algorithm: the sequence is split into chunks of size `Q = head_dim`, each processed by a CUDA intra-chunk scan kernel. Inter-chunk state propagation runs as a sequential scan over chunk summaries. An exact correction pass propagates inter-chunk states through the full intra-chunk dynamics (rotation + erasure), exploiting the linearity of the delta-rule in the initial state.

Adaptive Timestep
The timestep `dt` adapts to per-head dynamics: `dt_base = softplus(log_dt_scale) / (alpha + |omega| + eps)`, where the characteristic frequency `alpha + |omega|` normalizes the step size. An input-dependent adjustment `dt = dt_base + softplus(sel_dt)` allows the model to modulate temporal resolution per token. A smooth safety cap prevents instability in the low-frequency regime.

Variance-Preserving Initialization
Initialization follows T-Fixup depth scaling: projection weights scale as `1/sqrt(2 * n_layers)`, dynamics projections as `1/sqrt(n_layers)`, and depthwise convolution as `1/sqrt(kernel_size)`. Gate biases are set to `log(2)`, the unique value maximizing gradient flow through a product of two sigmoid gates (`d/d_sigma[sigma^2(1-sigma)] = 0` at `sigma = 2/3`).

Layer-Stratified Spectral Initialization
Each layer is assigned a band of the log-timescale range `[1, context_length]`. Early layers cover short timescales (local patterns), deep layers cover long timescales (document-level coherence). The band fraction `2/(n_layers + 1)` is the unique value giving exactly 50% overlap between adjacent layers. Per-head damping rates and frequencies within each band are initialized via inverse softplus from log-spaced timescales.

## Parameter-Free Configuration

The user specifies four values:

```python
config = CDSSMConfig(
    d_model=768,
    n_layers=12,
    context_length=1024,
    vocab_size=50257,
)
```

All other constants are derived:

| Constant | Value | Source |
|----------|-------|--------|
| `d_inner` | `2 * d_model` | Gate + content parallel pathways |
| `head_dim` | `64` | H100 Tensor Core tile width |
| `n_heads` | `d_inner / head_dim` | Structural |
| `chunk_size` | `head_dim` | SSD algorithm alignment |
| `conv_kernel_size` | `max(2, head_dim // 16)` | Sub-chunk local receptive field |
| `ssm_norm_groups` | `32` | GPU warp size |
| `rope_base` | `context_length` | Longest wavelength = sequence length |
| `gating_c` | `log(context_length)` | Signal decays to `1/L` over `L` steps |
| `spectral_band_fraction` | `2 / (n_layers + 1)` | 50% overlap between adjacent layers |
| Gate biases | `log(2)` | Max gradient of sigmoid product |
| `eps_norm` | `io_eps^2` | Gradient representability in BF16 |
| `eps_cayley_det` | `compute_eps` | FP32 machine epsilon (bounded denominator) |
| `eps_adaptive_dt` | `1.0` | Unit frequency floor |

## CUDA Kernels

All kernels use BF16 I/O with FP32 internal compute. Target: NVIDIA H100 (SM 9.0).

| Kernel | File | Description |
|--------|------|-------------|
| `dynamics_fused` | `dynamics_fused.cu` | Fused dynamics pipeline: parses 7 per-head gate scalars, computes softplus/RoPE/adaptive dt/Cayley/recurrence gate/VP scale/beta/sel_C in a single launch. Replaces ~15 elementwise PyTorch ops. |
| `normalize_kq` | `normalize_kq.cu` | Fused L2 normalization of K and Q vectors using warp-level reduction. |
| `conv1d_silu` | `conv1d_silu.cu` | Fused depthwise Conv1d + SiLU with causal padding. Eliminates transpose-conv-transpose-activation chain. |
| `intra_chunk_scan` | `intra_chunk_scan.cu` | Delta-rule scan within chunks. Register-resident `(2 x D)` state per thread column. Per timestep: retrieval, selective erasure, Cayley rotation, injection. Tracks cumulative `A_bar` products. |
| `inter_chunk_scan` | `inter_chunk_scan.cu` | Sequential state propagation between chunks: `state[k+1] = A[k] @ state[k] + h[k]`. |
| `cayley_vp` | `cayley_vp.cu` | Standalone Cayley discretization + recurrence gate + VP scale (subsumed by `dynamics_fused` in default path). |
| `adaptive_dt` | `adaptive_dt.cu` | Standalone adaptive timestep (subsumed by `dynamics_fused` in default path). |

The Cayley transform implementation in `cayley_math.cuh` uses FMA instructions (`__fmaf_rn`) and IEEE-compliant division (`__fdiv_rn`) for numerical precision.

## Installation

Requires: Python 3.10+, PyTorch 2.1+, CUDA 12.0+, NVIDIA H100 GPU.

```bash
pip install -e .
```

This compiles the CUDA extension `cdssm._C` targeting SM 9.0.

## Usage

### Training

```bash
# WikiText-103
python -m cdssm.train --config configs/default.yaml

# FineWeb-Edu (300M tokens, streaming)
python -m cdssm.train --config configs/fineweb.yaml
```

### Evaluation

```bash
# lm-evaluation-harness zero-shot benchmarks
python -m cdssm.eval --checkpoint checkpoints/best.pt \
    --tasks wikitext,lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande
```

### Hyperparameter Sweep

```bash
python -m cdssm.sweep --config configs/default.yaml \
    --lrs 6e-4 4e-4 --seed 42
```

### Inference

```python
from cdssm.inference.predict import load_model_from_checkpoint, generate_greedy

model, tokenizer = load_model_from_checkpoint("checkpoints/best.pt")
output = generate_greedy(model, tokenizer, "The meaning of", max_new_tokens=50)
```

## Training Configuration

The optimizer uses two parameter groups with depth-scaled learning rates:

| Group | Parameters | Learning Rate |
|-------|-----------|---------------|
| Standard | Projections, embeddings, norms | `base_lr` |
| SSM dynamics | `gate_proj`, `log_dt_scale` | `base_lr / sqrt(2 * n_layers)` |

Default training hyperparameters (`configs/training/default.yaml`):

```yaml
batch_size: 4
grad_accum_steps: 8       # effective batch = 32
base_lr: 6e-4
weight_decay: 0.1
betas: [0.9, 0.95]
warmup_steps: 500
precision: bfloat16
grad_clip: 1.0
min_lr_ratio: 0.0         # cosine decay to zero
```

## References

**Mamba-2: Transformers are SSMs** (Dao & Gu, 2024). [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
Establishes State Space Duality (SSD), showing structured SSMs can be computed as masked attention within chunks. CD-SSM adopts the chunkwise parallel scan algorithm from SSD, extending it to 2x2 Cayley dynamics with delta-rule state updates. The chunk size constraint `Q = head_dim` follows from the structured matrix decomposition in SSD Section 3.2.

**Gated Delta Networks** (Yang et al., 2024). [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
Introduces gated delta-rule memory with selective write/erase control for linear recurrences. CD-SSM implements the delta-rule update `h -= beta * (h @ k) k^T` with input-dependent gating `beta = sigmoid(beta_raw) * sigmoid(sel_B)`, and applies the exact correction scan exploiting linearity in the initial state.

**Griffin: Mixing Gated Linear Recurrences with Local Attention** (De et al., 2024). [arXiv:2402.19427](https://arxiv.org/abs/2402.19427)
Proposes the Real-Gated Linear Recurrent Unit (RG-LRU), where a learned gate modulates the recurrence eigenvalue per timestep. CD-SSM adapts this mechanism: the effective eigenvalue is `|eig|^{c * r}` where `r = sigmoid(r_gate)` and `c = log(context_length)`, allowing each head to interpolate between full retention (`r = 1/c`) and rapid decay (`r = 1`).

**S4: Structured State Spaces for Sequence Modeling** (Gu et al., 2022). [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)
Foundational work on structured state-space models for long-range dependencies. S4 introduces the HiPPO initialization and diagonal-plus-low-rank parameterization. CD-SSM replaces the HiPPO matrix with a Cayley-discretized 2x2 Hamiltonian (rotation-damping) system, trading the polynomial approximation basis for provable A-stability with explicit spectral control.

**Symplectic Recurrent Neural Networks** (Chen et al., 2020). [arXiv:1909.13334](https://arxiv.org/abs/1909.13334)
Demonstrates structure-preserving discretization for Hamiltonian dynamics in recurrent networks. CD-SSM extends this principle to the *dissipative* regime: the state matrix `A = [[-alpha, omega], [-omega, -alpha]]` is a damped Hamiltonian, and the Cayley transform preserves the mapping from the left half-plane to the unit disk. The variance-preserving scale `sqrt(1 - |eig_eff|^2)` ensures energy conservation at injection.
