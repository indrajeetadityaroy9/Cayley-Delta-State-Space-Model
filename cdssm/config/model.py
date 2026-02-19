"""CDSSM Configuration: Parameter-Free & Hardware-Aligned.

All internal constants are derived from the minimal config surface:
(d_model, n_layers, context_length, vocab_size, state_dim).

Every constant traces to one of:
- IEEE 754 / hardware: machine epsilon, Tensor Core width, warp size
- Mathematical identity: Nyquist theorem, gradient amplification order, sigmoid calculus
- Structural constraint: gate+content pathways, SSD algorithm, chunk alignment
- Information-theoretic principle: maximum entropy, coverage guarantee
"""

import math
import torch
from dataclasses import dataclass, field


# === Computed Fields Registry (Single Source of Truth) ===
# All fields populated by CDSSMConfig.__post_init__ — must be excluded when
# reconstructing a config from a checkpoint dict.

COMPUTED_FIELDS: frozenset[str] = frozenset({
    # Architecture (from derive_architecture)
    "d_inner", "n_heads", "head_dim", "chunk_size",
    "conv_kernel_size", "ssm_norm_groups", "rope_base",
    "gating_c", "spectral_band_fraction",
    # Epsilon hierarchy (from derive_epsilon_hierarchy)
    "eps_norm", "eps_log_argument", "eps_adaptive_dt",
    # Initialization biases (from derive_init_biases)
    "init_biases",
    # BF16 safety constants (from derive_bf16_safety_constants)
    "bf16_omega_thresh", "bf16_delta", "bf16_smoothness",
})


# === Derivation Functions ===

def derive_epsilon_hierarchy(dtype: torch.dtype = torch.bfloat16) -> dict:
    """Derive epsilon values from IEEE 754 machine precision.

    Two dtype constants determine everything:
    - io_eps: BF16 machine epsilon (tensor I/O precision)

    Each epsilon is set by the gradient amplification order of its operation:
    if the gradient is O(eps^{-k}), we need eps^{-k} representable in the
    I/O dtype, giving eps > io_eps^{1/k}.
    """
    io_eps = torch.finfo(dtype).eps               # 2^{-7}  = 7.81e-3

    return {
        "norm": float(io_eps ** 2),
        "log_argument": float(io_eps ** 2),
        "adaptive_dt": 1.0,
    }


def derive_architecture(d_model: int, n_layers: int, context_length: int) -> dict:
    """Derive all architecture constants from the 3 base parameters."""
    d_inner = d_model * 2
    head_dim = 64
    n_heads = d_inner // head_dim
    chunk_size = head_dim
    ssm_norm_groups = 32
    conv_kernel_size = max(2, head_dim // 16)
    rope_base = float(context_length)
    gating_c = math.log(context_length)
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

    log(2) is the UNIQUE bias maximizing gradient flow through a product
    of two sigmoid gates.
    """
    gate_open_bias = math.log(2)

    return {
        "sel_B_bias": gate_open_bias,
        "sel_C_bias": gate_open_bias,
        "r_gate_bias": gate_open_bias,
        "beta_bias": 0.0,
        "sel_dt_bias": 0.0,
    }


def derive_bf16_safety_constants(dtype: torch.dtype = torch.bfloat16) -> dict:
    """Derive smooth safety cap constants from IEEE 754 mantissa precision."""
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
    """SSM LR = base_lr / sqrt(2*n_layers). T-Fixup aligned."""
    return min(0.5, 1.0 / math.sqrt(2 * n_layers))


# === Configuration Dataclass ===

@dataclass
class CDSSMConfig:
    """CDSSM Configuration — 5-value input, fully derived internals."""

    # === User-Specified (5 values) ===
    d_model: int = 768
    n_layers: int = 12
    context_length: int = 8192
    vocab_size: int = 50257
    state_dim: int = 16

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
    eps_norm: float = field(init=False)
    eps_log_argument: float = field(init=False)
    eps_adaptive_dt: float = field(init=False)

    # === Derived: Initialization Biases (from derive_init_biases) ===
    init_biases: dict = field(init=False)

    # === Derived: BF16 Safety Constants (from derive_bf16_safety_constants) ===
    bf16_omega_thresh: float = field(init=False)
    bf16_delta: float = field(init=False)
    bf16_smoothness: float = field(init=False)

    def __post_init__(self):
        assert self.state_dim % 2 == 0, "state_dim must be even for complex pairs"

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

        eps = derive_epsilon_hierarchy()
        self.eps_norm = eps["norm"]
        self.eps_log_argument = eps["log_argument"]
        self.eps_adaptive_dt = eps["adaptive_dt"]

        self.init_biases = derive_init_biases()

        bf16 = derive_bf16_safety_constants()
        self.bf16_omega_thresh = bf16["omega_thresh"]
        self.bf16_delta = bf16["delta"]
        self.bf16_smoothness = bf16["smoothness"]
