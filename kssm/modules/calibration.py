"""Spectral calibration for KSSM."""

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
    power_accum = None
    total_samples = 0
    seq_len = None

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break

        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        # Canonical path expects token IDs and embedding to always be present.
        x = x.to(embedding.weight.device, non_blocking=True)
        x_embedded = embedding(x)
        x_float = x_embedded.float().mean(dim=-1)

        seq_len = x_float.shape[1]
        x_centered = x_float - x_float.mean(dim=1, keepdim=True)
        fft = torch.fft.rfft(x_centered, dim=1)
        power = (fft.abs() ** 2).mean(dim=0)

        if power_accum is None:
            power_accum = power
        else:
            power_accum = power_accum + power

        total_samples += 1

    power_avg = power_accum / total_samples
    freqs = torch.fft.rfftfreq(seq_len, d=1.0, device=power_avg.device)

    power_no_dc = power_avg[1:]
    freqs_no_dc = freqs[1:]
    cumsum = power_no_dc.cumsum(dim=0)
    cumsum = cumsum / cumsum[-1]

    idx_low = (cumsum >= percentile_low).nonzero(as_tuple=True)[0][0].item()
    idx_high = (cumsum >= percentile_high).nonzero(as_tuple=True)[0][0].item()

    freq_low = max(freqs_no_dc[idx_low].item(), 1.0 / (seq_len * 4))
    freq_high = max(freqs_no_dc[idx_high].item(), freq_low * 2)

    padding_factor = 2.0
    t_min = max(1.0, 1.0 / (freq_high * padding_factor))
    t_max = min(float(seq_len) * 4, 1.0 / (freq_low / padding_factor))
    t_max = max(t_max, t_min * 10)

    return {
        "t_min": t_min,
        "t_max": t_max,
        "freq_min": freq_low / padding_factor,
        "freq_max": freq_high * padding_factor,
    }
