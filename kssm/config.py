"""KSSM Configuration for Kinetic State Space Model."""

import math
from dataclasses import dataclass, field, asdict

import torch


def _largest_divisor_leq(n: int, k: int) -> int:
    """Return the largest divisor of n that is <= k."""
    for d in range(k, 0, -1):
        if n % d == 0:
            return d
    return 1


def bf16_safety_constants(dtype: torch.dtype) -> tuple[float, float, float]:
    """Derive smooth safety cap constants from dtype mantissa precision.

    For bf16 (7 mantissa bits, eps=7.81e-3):
        omega_thresh = sqrt(eps) ≈ 0.088
        delta = 16 * eps ≈ 0.125
        smoothness = omega_thresh / 5 ≈ 0.018

    For fp32 (23 mantissa bits, eps=1.19e-7):
        omega_thresh ≈ 3.4e-4 (cap essentially never engages)

    Returns:
        (omega_thresh, delta, smoothness)
    """
    eps = torch.finfo(dtype).eps
    omega_thresh = math.sqrt(eps)
    delta = 16.0 * eps
    smoothness = omega_thresh / 5.0
    return omega_thresh, delta, smoothness


def dynamics_scale(n_layers: int) -> float:
    """Derive dynamics projection dampening from depth.

    Output perturbation magnitude = dynamics_scale / sqrt(2 * n_layers),
    which gives O(1/n_layers) scaling — keeping weight perturbations
    small relative to spectral init biases.
    """
    return 1.0 / math.sqrt(n_layers)


@dataclass
class KSSMConfig:
    """KSSM configuration.

    Only d_model and n_layers are required. All other architectural
    parameters are derived from these two values using principled formulas.

    Optional calibration via calibrate_spectral_bounds() refines the
    spectral initialization for task-specific data.

    Usage (minimal, no data required):
        config = KSSMConfig(d_model=512, n_layers=8)
        model = KSSMLMHeadModel(config, vocab_size=50257)

    Usage (with calibration for best accuracy):
        config = KSSMConfig(d_model=512, n_layers=8)
        bounds = calibrate_spectral_bounds(dataloader, config.d_model)
        config = config.with_calibration(**bounds)
        model = KSSMLMHeadModel(config, vocab_size=50257)

    Constraints: d_inner must be even and divisible by n_heads.
    """

    # === Required ===
    d_model: int = 768
    n_layers: int = 12

    # === Optional overrides (derived if None) ===
    d_inner: int | None = None            # Default: 2 * d_model
    n_heads: int | None = None            # Default: max(8, d_inner // 48)

    # === Spectral Bounds (optional calibration override) ===
    calibrated_t_min: float | None = None
    calibrated_t_max: float | None = None
    calibrated_freq_min: float | None = None
    calibrated_freq_max: float | None = None

    # === Computed in __post_init__ (never set by user) ===
    head_dim: int = field(init=False)
    ssm_norm_groups: int = field(init=False)

    @property
    def t_min(self) -> float:
        """Minimum timescale. One-step timescale (universal minimum)."""
        return self.calibrated_t_min if self.calibrated_t_min is not None else 1.0

    @property
    def t_max(self) -> float:
        """Maximum timescale. Scales with n_heads capacity."""
        if self.calibrated_t_max is not None:
            return self.calibrated_t_max
        return float(max(self.n_heads ** 2, 1000))

    @property
    def freq_min(self) -> float:
        """Minimum frequency. Nyquist / capacity."""
        if self.calibrated_freq_min is not None:
            return self.calibrated_freq_min
        return 0.5 / max(self.n_heads, 64)

    @property
    def freq_max(self) -> float:
        """Maximum frequency. Nyquist frequency (0.5)."""
        return self.calibrated_freq_max if self.calibrated_freq_max is not None else 0.5

    def __post_init__(self):
        if self.d_inner is None:
            self.d_inner = self.d_model * 2

        if self.n_heads is None:
            self.n_heads = max(8, self.d_inner // 48)

        if self.d_inner % self.n_heads != 0:
            raise ValueError(f"d_inner ({self.d_inner}) must be divisible by n_heads ({self.n_heads})")

        self.head_dim = self.d_inner // self.n_heads

        if self.d_inner % 2 != 0:
            raise ValueError(f"d_inner must be even, got {self.d_inner}")

        # Derive GroupNorm groups: largest divisor of d_inner <= 32 (Wu & He, 2018)
        self.ssm_norm_groups = _largest_divisor_leq(self.d_inner, 32)

        # Enforce t_min < t_max if both provided
        if self.calibrated_t_min is not None and self.calibrated_t_max is not None:
            if self.calibrated_t_min >= self.calibrated_t_max:
                raise ValueError(f"t_min ({self.calibrated_t_min}) must be < t_max ({self.calibrated_t_max})")

    def with_calibration(
        self,
        t_min: float,
        t_max: float,
        freq_min: float,
        freq_max: float,
    ) -> "KSSMConfig":
        """Return a new config with calibrated spectral bounds.

        Usage:
            bounds = calibrate_spectral_bounds(dataloader, d_model)
            config = config.with_calibration(**bounds)
        """
        return KSSMConfig(
            d_model=self.d_model,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            calibrated_t_min=t_min,
            calibrated_t_max=t_max,
            calibrated_freq_min=freq_min,
            calibrated_freq_max=freq_max,
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization.

        Excludes computed fields.
        """
        d = asdict(self)
        for key in ("head_dim", "ssm_norm_groups"):
            d.pop(key, None)
        return d

