"""Spectral calibration for SC-KSSM (Self-Calibrating KSSM).

This module provides the data-driven calibration function that analyzes
input data to derive optimal timescale and frequency bounds for spectral
initialization. This eliminates the need for manual hyperparameter tuning.

Usage:
    bounds = calibrate_spectral_bounds(dataloader, d_model=512)
    config = KSSMConfig(
        calibrated_t_min=bounds["t_min"],
        calibrated_t_max=bounds["t_max"],
        calibrated_freq_min=bounds["freq_min"],
        calibrated_freq_max=bounds["freq_max"],
    )
"""

import torch


@torch.no_grad()
def calibrate_spectral_bounds(
    dataloader,
    d_model: int,
    n_batches: int = 50,
    percentile_low: float = 0.05,
    percentile_high: float = 0.95,
    embedding: torch.nn.Module | None = None,
) -> dict:
    """
    One-time calibration function. Run BEFORE model creation.

    Analyzes the frequency content of input data to derive optimal
    timescale and frequency bounds for spectral initialization.

    Args:
        dataloader: DataLoader yielding (x, y) or (x, y, mask) batches
        d_model: Model dimension (for embedding if needed)
        n_batches: Number of batches to analyze (default 50 for robust estimation)
        percentile_low: Lower percentile for frequency bounds (default 5%)
        percentile_high: Upper percentile for frequency bounds (default 95%)
        embedding: Optional embedding layer for converting token IDs to vectors.
                   When provided, token IDs are embedded before FFT analysis
                   instead of using the crude positional proxy.

    Returns:
        dict with keys: t_min, t_max, freq_min, freq_max
    """
    power_accum = None
    total_samples = 0
    seq_len = None

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break

        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        # x should be (batch, seq_len) for tokens or (batch, seq_len, d) for embeddings
        if x.dim() == 2:
            if embedding is not None:
                # Use provided embedding for proper frequency analysis
                x_embedded = embedding(x)  # (batch, seq_len, d_model)
                x_float = x_embedded.float().mean(dim=-1)  # Reduce to (batch, seq)
            else:
                # Token IDs - create simple positional encoding for frequency analysis
                batch_size, seq_len_curr = x.shape
                positions = torch.arange(seq_len_curr, device=x.device, dtype=torch.float32)
                positions = positions.unsqueeze(0).expand(batch_size, -1)
                x_float = positions + x.float() * (1.0 / seq_len_curr)
        else:
            # Already embeddings
            x_float = x.float().mean(dim=-1)  # Reduce to (batch, seq)

        seq_len = x_float.shape[1]

        # Center the signal
        x_centered = x_float - x_float.mean(dim=1, keepdim=True)

        # Compute FFT
        fft = torch.fft.rfft(x_centered, dim=1)
        power = (fft.abs() ** 2).mean(dim=0)  # Average over batch

        if power_accum is None:
            power_accum = power
        else:
            power_accum = power_accum + power

        total_samples += 1

    if power_accum is None or seq_len is None:
        raise ValueError(
            "Calibration requires data. The dataloader yielded no batches. "
            "Ensure n_batches > 0 and the dataloader is not empty."
        )

    # Average power spectrum
    power_avg = power_accum / total_samples

    # Frequency bins (normalized to [0, 0.5] Nyquist)
    freqs = torch.fft.rfftfreq(seq_len, d=1.0, device=power_avg.device)

    # Compute cumulative distribution for percentiles
    # Skip DC component (index 0)
    power_no_dc = power_avg[1:]
    freqs_no_dc = freqs[1:]

    if power_no_dc.sum() < 1e-10:
        # Flat spectrum - use sequence length as guide
        return {
            "t_min": 1.0,
            "t_max": float(seq_len) * 2,
            "freq_min": 1.0 / (seq_len * 2),
            "freq_max": 0.5,  # Nyquist
        }

    # Normalize to get CDF
    cumsum = power_no_dc.cumsum(dim=0)
    cumsum = cumsum / cumsum[-1]

    # Find percentile frequencies
    idx_low = (cumsum >= percentile_low).nonzero(as_tuple=True)[0]
    idx_high = (cumsum >= percentile_high).nonzero(as_tuple=True)[0]

    if len(idx_low) == 0:
        idx_low = 0
    else:
        idx_low = idx_low[0].item()

    if len(idx_high) == 0:
        idx_high = len(freqs_no_dc) - 1
    else:
        idx_high = idx_high[0].item()

    freq_low = max(freqs_no_dc[idx_low].item(), 1.0 / (seq_len * 4))
    freq_high = max(freqs_no_dc[idx_high].item(), freq_low * 2)

    # Convert to timescales: t = 1/f
    # Add padding factor for safety
    padding_factor = 2.0
    t_min = max(1.0, 1.0 / (freq_high * padding_factor))
    t_max = min(float(seq_len) * 4, 1.0 / (freq_low / padding_factor))

    # Ensure reasonable bounds
    t_max = max(t_max, t_min * 10)
    freq_min_out = freq_low / padding_factor
    freq_max_out = freq_high * padding_factor

    # Validate bounds
    assert t_min < t_max, f"Calibration failed: t_min ({t_min}) >= t_max ({t_max})"
    assert freq_min_out < freq_max_out, f"Calibration failed: freq_min ({freq_min_out}) >= freq_max ({freq_max_out})"

    return {
        "t_min": t_min,
        "t_max": t_max,
        "freq_min": freq_min_out,
        "freq_max": freq_max_out,
    }
