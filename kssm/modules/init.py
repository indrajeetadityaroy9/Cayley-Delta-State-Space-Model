"""Nuclear initialization for KSSM layers.

SSMs are famously sensitive to initialization. KSSM relies on "Nuclear Initialization"
which starts parameters near the critical boundary (unit circle eigenvalues).

Key insight from plan:
- softplus(-2.0) ≈ 0.13 → 0.87^200 ≈ 10^-13 (signal vanishes!)
- softplus(-5.0) ≈ 0.007 → 0.993^500 ≈ 0.03 (signal retained)

For long-memory tasks like induction head, we need near-zero damping.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


def _get_projection_info(layer: nn.Module) -> dict:
    """Get projection parameters from a layer (handles both fused and separate).

    Returns dict with keys: 'is_fused', 'd_inner', and projection references.
    """
    info = {'is_fused': False, 'd_inner': None}

    # Check for fused projection (kssm_proj)
    if hasattr(layer, 'projections') and hasattr(layer.projections, 'kssm_proj'):
        proj = layer.projections
        info['is_fused'] = True
        info['kssm_proj'] = proj.kssm_proj
        info['d_inner'] = proj.d_inner
        if hasattr(proj, 'in_proj'):
            info['in_proj'] = proj.in_proj
    elif hasattr(layer, 'kssm_proj'):
        info['is_fused'] = True
        info['kssm_proj'] = layer.kssm_proj
        info['d_inner'] = layer.d_model
    # Check for separate projections
    elif hasattr(layer, 'projections') and hasattr(layer.projections, 'alpha_proj'):
        proj = layer.projections
        info['alpha_proj'] = proj.alpha_proj
        info['omega_proj'] = proj.omega_proj
        info['B_proj'] = proj.B_proj
        info['d_inner'] = proj.d_inner
    elif hasattr(layer, 'alpha_proj'):
        info['alpha_proj'] = layer.alpha_proj
        info['omega_proj'] = layer.omega_proj
        info['B_proj'] = layer.B_proj
        info['d_inner'] = layer.d_model

    # C projection (output)
    if hasattr(layer, 'output_proj') and hasattr(layer.output_proj, 'C_proj'):
        info['C_proj'] = layer.output_proj.C_proj
    elif hasattr(layer, 'C_proj'):
        info['C_proj'] = layer.C_proj

    return info


def nuclear_init(
    layer: nn.Module,
    long_memory: bool = True,
    freq_min: float = 0.01,
    freq_max: float = 100.0,
) -> None:
    """Apply nuclear initialization to a KSSM layer.

    Initializes parameters near the critical boundary (unit circle eigenvalues)
    for optimal long-range memory retention.

    Supports both fused (kssm_proj) and separate (alpha_proj, omega_proj, etc.)
    projection architectures.

    Args:
        layer: KSSMLayer or KSSMLayerSimple to initialize.
        long_memory: If True, use very low damping for long-range tasks.
                     If False, use moderate damping.
        freq_min: Minimum frequency for log-uniform distribution.
        freq_max: Maximum frequency for log-uniform distribution.
    """
    # Alpha bias: controls damping
    # softplus(bias) ≈ damping rate
    # For long memory: use -5.0 (softplus ≈ 0.007)
    # For moderate memory: use -2.0 (softplus ≈ 0.13)
    alpha_bias_val = -5.0 if long_memory else -2.0

    info = _get_projection_info(layer)
    d_inner = info.get('d_inner')

    if d_inner is None:
        return  # Not a KSSM layer we recognize

    # Log-uniform frequencies
    log_freq_min = math.log(freq_min)
    log_freq_max = math.log(freq_max)
    freqs = torch.exp(torch.linspace(log_freq_min, log_freq_max, d_inner))

    if info['is_fused']:
        # Fused projection layout: [alpha, omega, B, dt] = [d, d, 2d, d] = 5d
        kssm_proj = info['kssm_proj']
        d = d_inner

        with torch.no_grad():
            # Alpha: zero weight, constant bias
            kssm_proj.weight[:d, :].zero_()
            kssm_proj.bias[:d].fill_(alpha_bias_val)

            # Omega: zero weight, log-uniform bias
            kssm_proj.weight[d:2*d, :].zero_()
            kssm_proj.bias[d:2*d].copy_(freqs)

            # B: small random weight, zero bias
            nn.init.normal_(kssm_proj.weight[2*d:4*d, :], std=0.02)
            kssm_proj.bias[2*d:4*d].zero_()

            # dt: keep existing initialization (log-uniform already set)
    else:
        # Separate projections
        if 'alpha_proj' in info:
            alpha_proj = info['alpha_proj']
            nn.init.zeros_(alpha_proj.weight)
            nn.init.constant_(alpha_proj.bias, alpha_bias_val)

        if 'omega_proj' in info:
            omega_proj = info['omega_proj']
            with torch.no_grad():
                nn.init.zeros_(omega_proj.weight)
                omega_proj.bias.copy_(freqs)

        if 'B_proj' in info:
            B_proj = info['B_proj']
            nn.init.normal_(B_proj.weight, std=0.02)
            nn.init.zeros_(B_proj.bias)

    # C projection: small random initialization
    if 'C_proj' in info:
        C_proj = info['C_proj']
        nn.init.normal_(C_proj.weight, std=0.02)
        nn.init.zeros_(C_proj.bias)


def hippo_init(
    layer: nn.Module,
    method: str = "legs",
) -> None:
    """Apply HiPPO-style initialization (experimental).

    HiPPO (High-order Polynomial Projection Operators) initialization
    from the S4 paper. This provides theoretically optimal memory.

    Supports both fused and separate projection architectures.

    Args:
        layer: KSSM layer to initialize.
        method: HiPPO variant ('legs', 'legt', 'lagt').
    """
    info = _get_projection_info(layer)
    d_inner = info.get('d_inner')

    if d_inner is None:
        return

    # HiPPO-LegS frequencies (approximation)
    # These are based on Legendre polynomial roots
    if method == "legs":
        n = torch.arange(1, d_inner + 1, dtype=torch.float32)
        freqs = torch.sqrt(n * (n + 1))
    elif method == "legt":
        n = torch.arange(1, d_inner + 1, dtype=torch.float32)
        freqs = 2 * n + 1
    else:
        freqs = torch.exp(torch.linspace(-2, 2, d_inner))

    alpha_bias_val = -5.0  # Very low damping for HiPPO

    if info['is_fused']:
        kssm_proj = info['kssm_proj']
        d = d_inner

        with torch.no_grad():
            # Alpha: zero weight, constant bias (low damping)
            kssm_proj.weight[:d, :].zero_()
            kssm_proj.bias[:d].fill_(alpha_bias_val)

            # Omega: zero weight, HiPPO frequencies
            kssm_proj.weight[d:2*d, :].zero_()
            kssm_proj.bias[d:2*d].copy_(freqs)
    else:
        if 'omega_proj' in info:
            omega_proj = info['omega_proj']
            with torch.no_grad():
                nn.init.zeros_(omega_proj.weight)
                omega_proj.bias.copy_(freqs)

        if 'alpha_proj' in info:
            alpha_proj = info['alpha_proj']
            nn.init.zeros_(alpha_proj.weight)
            nn.init.constant_(alpha_proj.bias, alpha_bias_val)


def verify_initialization(layer: nn.Module) -> dict:
    """Verify initialization parameters and expected behavior.

    Returns a dictionary with initialization diagnostics.
    Supports both fused and separate projection architectures.

    Args:
        layer: KSSM layer to verify.

    Returns:
        Dictionary with:
        - alpha_bias: Current alpha bias value
        - expected_damping: Expected damping from softplus
        - decay_100: Expected signal after 100 steps
        - decay_500: Expected signal after 500 steps
        - freq_min/max/mean: Range of initialized frequencies
    """
    import torch.nn.functional as F

    diagnostics = {}
    info = _get_projection_info(layer)
    d_inner = info.get('d_inner')

    if d_inner is None:
        return diagnostics

    if info['is_fused']:
        # Fused projection layout: [alpha, omega, B, dt] = [d, d, 2d, d]
        kssm_proj = info['kssm_proj']
        d = d_inner

        # Alpha bias (first d elements)
        alpha_bias = kssm_proj.bias[:d].mean().item()
        expected_damping = F.softplus(torch.tensor(alpha_bias)).item()
        decay_factor = 1.0 - expected_damping

        diagnostics['alpha_bias'] = alpha_bias
        diagnostics['expected_damping'] = expected_damping
        diagnostics['decay_100'] = decay_factor ** 100
        diagnostics['decay_500'] = decay_factor ** 500

        # Omega range (elements d to 2d)
        freqs = kssm_proj.bias[d:2*d].detach()
        diagnostics['freq_min'] = freqs.min().item()
        diagnostics['freq_max'] = freqs.max().item()
        diagnostics['freq_mean'] = freqs.mean().item()
    else:
        # Separate projections
        if 'alpha_proj' in info:
            alpha_proj = info['alpha_proj']
            alpha_bias = alpha_proj.bias.mean().item()
            expected_damping = F.softplus(torch.tensor(alpha_bias)).item()
            decay_factor = 1.0 - expected_damping

            diagnostics['alpha_bias'] = alpha_bias
            diagnostics['expected_damping'] = expected_damping
            diagnostics['decay_100'] = decay_factor ** 100
            diagnostics['decay_500'] = decay_factor ** 500

        if 'omega_proj' in info:
            omega_proj = info['omega_proj']
            freqs = omega_proj.bias.detach()
            diagnostics['freq_min'] = freqs.min().item()
            diagnostics['freq_max'] = freqs.max().item()
            diagnostics['freq_mean'] = freqs.mean().item()

    return diagnostics


def init_kssm_model(model: nn.Module, long_memory: bool = True) -> None:
    """Initialize all KSSM layers in a model.

    Recursively finds and initializes all KSSM layers.

    Args:
        model: Model containing KSSM layers.
        long_memory: Whether to use long-memory initialization.
    """
    from kssm.modules.kssm_layer import KSSMLayer, KSSMLayerSimple

    for module in model.modules():
        if isinstance(module, (KSSMLayer, KSSMLayerSimple)):
            nuclear_init(module, long_memory=long_memory)
