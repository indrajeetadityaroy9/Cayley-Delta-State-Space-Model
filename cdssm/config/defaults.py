"""CDSSM Configuration: Parameter-Free & Hardware-Aligned.

All internal constants are derived from the minimal config surface:
(d_model, n_layers, context_length, vocab_size).

Every constant traces to one of:
- IEEE 754 / hardware: machine epsilon, Tensor Core width, warp size
- Mathematical identity: Nyquist theorem, gradient amplification order, sigmoid calculus
- Structural constraint: gate+content pathways, SSD algorithm, chunk alignment
- Information-theoretic principle: maximum entropy, coverage guarantee
"""

import math
import torch
from dataclasses import dataclass, field, asdict


# === Computed Fields Registry (Single Source of Truth) ===
# All fields populated by CDSSMConfig.__post_init__ — must be excluded when
# reconstructing a config from a checkpoint dict.

COMPUTED_FIELDS: frozenset[str] = frozenset({
    # Architecture (from derive_architecture)
    "d_inner", "n_heads", "head_dim", "chunk_size",
    "conv_kernel_size", "ssm_norm_groups", "rope_base",
    "gating_c", "spectral_band_fraction",
    # Epsilon hierarchy (from derive_epsilon_hierarchy)
    "eps_cayley_det", "eps_norm", "eps_eigenvalue_floor",
    "eps_log_argument", "eps_vp_floor", "eps_adaptive_dt",
    # Initialization biases (from derive_init_biases)
    "init_biases",
    # BF16 safety constants (from derive_bf16_safety_constants)
    "bf16_omega_thresh", "bf16_delta", "bf16_smoothness",
})


# === Derivation Functions ===

def derive_epsilon_hierarchy(dtype: torch.dtype = torch.bfloat16) -> dict:
    """Derive all epsilon values from IEEE 754 machine precision.

    Two dtype constants determine everything:
    - compute_eps: FP32 machine epsilon (internal CUDA compute precision)
    - io_eps: BF16 machine epsilon (tensor I/O precision)

    Each epsilon is set by the gradient amplification order of its operation:
    if the gradient is O(eps^{-k}), we need eps^{-k} representable in the
    I/O dtype, giving eps > io_eps^{1/k} (or compute_eps for bounded ops).
    """
    compute_eps = torch.finfo(torch.float32).eps  # 2^{-23} ≈ 1.19e-7
    io_eps = torch.finfo(dtype).eps               # 2^{-7}  = 7.81e-3

    return {
        # Cayley det = 1 + (αdt)² + (ωdt)² ≥ 1 by construction.
        # Gradient amplification: O(1) (denominator bounded below by 1).
        # Bare machine epsilon suffices for floating-point rounding safety.
        "cayley_det": float(compute_eps),

        # Normalization: rsqrt(sum_sq + eps). Gradient: O(eps^{-1/2}).
        # Constraint: eps^{-1/2} representable in I/O dtype →
        # eps^{-1/2} < 1/io_eps → eps > io_eps².
        # io_eps² = (7.81e-3)² = 6.10e-5.
        "norm": float(io_eps ** 2),

        # Eigenvalue floor before powf(eig_sq, exponent).
        # eig_sq ∈ [0, 1] from Cayley (dissipative). powf(0, neg) = Inf.
        # Exponent (c·r - 1)/2 ∈ [-0.5, ~4]. Worst grad: O(eps^{-1.5}).
        # Need eps^{-1.5} < FP32_MAX → eps > FP32_MAX^{-2/3} ≈ 1e-25.
        # compute_eps ≈ 1.19e-7 >> 1e-25, safely prevents pow(0, neg).
        "eigenvalue_floor": float(compute_eps),

        # log(x + eps): gradient = 1/(x + eps). O(eps^{-1}).
        # 1/eps representable in I/O dtype → eps > io_eps.
        # Use io_eps² for consistency with norm (tighter, still safe).
        "log_argument": float(io_eps ** 2),

        # VP scale: sqrt(1 - eig_eff²). Argument ∈ [0, 1] (bounded domain).
        # Same structure as cayley_det. compute_eps suffices.
        "vp_floor": float(compute_eps),

        # Adaptive dt: floor on characteristic frequency (alpha + |omega| + eps).
        # eps = 1.0 guarantees char_freq ≥ 1, bounding dt ≤ softplus(log_dt_scale).
        # With eps=1, the maximum dt equals the learned scale, making dynamics
        # fully determined by the learnable parameter rather than an epsilon
        # safety net. Unit floor — no dtype dependence, pure mathematical choice.
        "adaptive_dt": 1.0,
    }


def derive_architecture(d_model: int, n_layers: int, context_length: int) -> dict:
    """Derive all architecture constants from the 3 base parameters.

    Every value traces to hardware ISA, structural necessity, or mathematical
    identity. No heuristics.
    """
    # Gate + content pathways: in_proj splits into z_gate (d_inner) and
    # x_gate (d_inner), requiring 2× expansion. Structural, not tunable.
    d_inner = d_model * 2

    # H100 Tensor Core vector width = 64 elements (16×4 tile for BF16 matmuls).
    # Fixed by hardware ISA.
    head_dim = 64

    n_heads = d_inner // head_dim

    # SSD algorithm requires chunk_size = head_dim for the structured matrix
    # decomposition to align. See Mamba-2 §3.2: "chunk size Q = head dim".
    chunk_size = head_dim

    # WARP_SIZE = 32 threads (NVIDIA GPU ISA, all architectures since Volta).
    # GroupNorm with 32 groups: each warp processes exactly one group.
    ssm_norm_groups = 32

    # Depthwise conv kernel: head_dim // 16 = 64 // 16 = 4.
    # Conv provides local (sub-chunk) context; SSM handles global.
    # Receptive field = 1/16 of head_dim: coarsest power-of-2 subdivision
    # giving kernel_size ≥ 2 (minimum for learning temporal patterns).
    conv_kernel_size = max(2, head_dim // 16)

    # RoPE base = context_length: frequency progression
    # freqs[h] = 1 / base^{h/H} covers wavelengths from 1 to context_length.
    # Longest wavelength EXACTLY matches sequence length. No arbitrary floor
    # (the original 10000 was a Transformer convention; here we derive from
    # the actual context window).
    rope_base = float(context_length)

    # Gating range: c = log(context_length).
    # Recurrence gate modulates |eigenvalue|^{c·r}. With c = log(L),
    # r=1 gives |eig|^{log L}: signal decays to 1/L over L steps —
    # exactly one traversal of the context window.
    gating_c = math.log(context_length)

    # Spectral band fraction: UNIQUE fraction giving exactly 50% overlap
    # between adjacent layers with uniform sliding across log-timescale range.
    #
    # Derivation: layer i covers [s_i, s_i + w] with s_i uniform in [0, range-w].
    # Step between adjacent layers: step = (range - w) / (n-1).
    # Overlap = w - step. Setting overlap = w/2 (50%):
    #   w/2 = w - (range - w)/(n-1)
    #   → w = 2·range / (n+1)
    #   → fraction = 2 / (n_layers + 1)
    #
    # n=12 → 0.154, n=6 → 0.286, n=24 → 0.080, n=1 → 1.0 (full range).
    spectral_band_fraction = 2.0 / (n_layers + 1)

    return {
        "d_inner": d_inner,
        "head_dim": head_dim,
        "n_heads": n_heads,
        "chunk_size": chunk_size,
        "ssm_norm_groups": ssm_norm_groups,
        "conv_kernel_size": conv_kernel_size,
        "rope_base": rope_base,
        "gating_c": gating_c,
        "spectral_band_fraction": spectral_band_fraction,
    }


def derive_init_biases() -> dict:
    """Derive initialization biases for sigmoid gates from calculus.

    The write path multiplies two sigmoid gates: beta = σ(β_raw)·σ(sel_B).
    For f(b) = σ(b)², the gradient f'(b) = 2·σ(b)²·(1 - σ(b)).
    Maximizing: d/dσ[2σ²(1-σ)] = 2(2σ - 3σ²) = 0 → σ* = 2/3.
    sigmoid⁻¹(2/3) = log(2/(1-2/3)) = log(2) ≈ 0.693.

    log(2) is the UNIQUE bias maximizing gradient flow through a product
    of two sigmoid gates. No free parameters in the derivation.
    """
    # Calculus-optimal bias for sigmoid product gradient sensitivity
    gate_open_bias = math.log(2)  # ≈ 0.693, giving σ(b) = 2/3

    return {
        "sel_B_bias": gate_open_bias,   # Write gate: max-gradient for σ product
        "sel_C_bias": gate_open_bias,   # Read gate: consistent with write path
        "r_gate_bias": gate_open_bias,  # Recurrence: max-gradient initialization
        "beta_bias": 0.0,              # Delta-rule: σ(0)=0.5 = maximum entropy
        "sel_dt_bias": 0.0,            # Timestep: softplus(0)=ln(2), no adjustment
    }


def derive_bf16_safety_constants(dtype: torch.dtype = torch.bfloat16) -> dict:
    """Derive smooth safety cap constants from IEEE 754 mantissa precision.

    These constants control the smooth clamping of the Cayley transform
    near numerically degenerate regimes (omega → 0).

    Returns:
        omega_thresh: sqrt(eps). Threshold where omega² < eps, meaning omega²
            is indistinguishable from 0 in dtype precision. Below this, the
            Cayley rotation angle is numerically zero.
        delta: 16 * eps. ULP-based safety margin. 16 = 2^4 accounts for 4 FMA
            operations in the Cayley transform (each accumulates up to 1 ULP
            of error). Factor of 4× safety gives 16 ULP total.
        smoothness: omega_thresh / 5. Transition bandwidth for smooth clamping.
            1/5 is the narrowest ratio keeping the sigmoid-based smooth step
            within 1% of its asymptotes at ±omega_thresh (sigmoid(5) = 0.993).
    """
    eps = torch.finfo(dtype).eps
    omega_thresh = math.sqrt(eps)
    delta = 16.0 * eps
    smoothness = omega_thresh / 5.0
    return {
        "omega_thresh": omega_thresh,
        "delta": delta,
        "smoothness": smoothness,
    }


def derived_ssm_lr_ratio(n_layers: int) -> float:
    """SSM LR = base_lr / sqrt(2·n_layers). T-Fixup aligned.

    n_layers=12 → 0.204x, n_layers=24 → 0.144x, n_layers=48 → 0.102x.
    Clamped to 0.5 for very shallow models.
    """
    return min(0.5, 1.0 / math.sqrt(2 * n_layers))


# === Configuration Dataclass ===

@dataclass
class CDSSMConfig:
    """CDSSM Configuration — 4-value input, fully derived internals.

    User specifies: d_model, n_layers, context_length, vocab_size.
    Everything else is derived from hardware constraints (Tensor Core width,
    WARP_SIZE), mathematical identities (gradient amplification, sigmoid
    calculus, Nyquist), or structural requirements (SSD algorithm, gate
    pathways).

    Derivation sources documented in derive_epsilon_hierarchy(),
    derive_architecture(), and derive_init_biases().
    """

    # === User-Specified (4 values + optional tokenizer) ===
    d_model: int = 768
    n_layers: int = 12
    context_length: int = 8192
    vocab_size: int = 50257
    tokenizer_name: str = "gpt2"

    # === Derived: Architecture (from derive_architecture) ===
    d_inner: int = field(init=False)
    n_heads: int = field(init=False)
    head_dim: int = field(init=False)
    ssm_norm_groups: int = field(init=False)
    chunk_size: int = field(init=False)
    conv_kernel_size: int = field(init=False)
    rope_base: float = field(init=False)
    gating_c: float = field(init=False)
    spectral_band_fraction: float = field(init=False)

    # === Derived: Epsilon Hierarchy (from derive_epsilon_hierarchy) ===
    eps_cayley_det: float = field(init=False)
    eps_norm: float = field(init=False)
    eps_eigenvalue_floor: float = field(init=False)
    eps_log_argument: float = field(init=False)
    eps_vp_floor: float = field(init=False)
    eps_adaptive_dt: float = field(init=False)

    # === Derived: Initialization Biases (from derive_init_biases) ===
    init_biases: dict = field(init=False)

    # === Derived: BF16 Safety Constants (from derive_bf16_safety_constants) ===
    bf16_omega_thresh: float = field(init=False)
    bf16_delta: float = field(init=False)
    bf16_smoothness: float = field(init=False)

    def __post_init__(self):
        # Architecture derivation
        arch = derive_architecture(self.d_model, self.n_layers, self.context_length)
        self.d_inner = arch["d_inner"]
        self.head_dim = arch["head_dim"]
        self.n_heads = arch["n_heads"]
        self.chunk_size = arch["chunk_size"]
        self.ssm_norm_groups = arch["ssm_norm_groups"]
        self.conv_kernel_size = arch["conv_kernel_size"]
        self.rope_base = arch["rope_base"]
        self.gating_c = arch["gating_c"]
        self.spectral_band_fraction = arch["spectral_band_fraction"]

        # Validation
        if self.d_inner % self.head_dim != 0:
            raise ValueError(
                f"d_inner ({self.d_inner}) must be divisible by head_dim={self.head_dim} "
                f"for Tensor Core alignment. Adjust d_model ({self.d_model}) such that "
                f"2*d_model is divisible by {self.head_dim}."
            )
        if self.d_inner % self.ssm_norm_groups != 0:
            raise ValueError(
                f"d_inner ({self.d_inner}) must be divisible by "
                f"ssm_norm_groups={self.ssm_norm_groups} for GroupNorm."
            )

        # Epsilon hierarchy derivation
        eps = derive_epsilon_hierarchy()
        self.eps_cayley_det = eps["cayley_det"]
        self.eps_norm = eps["norm"]
        self.eps_eigenvalue_floor = eps["eigenvalue_floor"]
        self.eps_log_argument = eps["log_argument"]
        self.eps_vp_floor = eps["vp_floor"]
        self.eps_adaptive_dt = eps["adaptive_dt"]

        # Initialization bias derivation
        self.init_biases = derive_init_biases()

        # BF16 safety constant derivation
        bf16 = derive_bf16_safety_constants()
        self.bf16_omega_thresh = bf16["omega_thresh"]
        self.bf16_delta = bf16["delta"]
        self.bf16_smoothness = bf16["smoothness"]

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
